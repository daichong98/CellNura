"""Portable baseline contracts and negative identity tests; no model training."""
import contextlib,hashlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from cellnura_release import baselines
from cellnura_release.baseline_runtime import resource,frozen_roles
from cellnura_release.paths import load_config,project_root

class BaselineReleaseTests(unittest.TestCase):
    def setUp(self):self.cfg=load_config()
    def test_catalog_no_contaminated_reference_in_A(self):
        for name in ['cellvit_sam','cellvitpp','hovernet_tphead','hovernet_official']:
            spec=baselines.catalog()['methods'][name]
            self.assertEqual(set(spec['tier'].values()),{'B'})
            self.assertEqual(spec['actions'],['infer'])
        self.assertEqual(baselines.catalog()['methods']['micronet']['tier']['consep'],'A')
        self.assertEqual(set(baselines.catalog()['methods']['cellnura_cellvit']['tier'].values()),{'C'})
        self.assertNotIn('cellvit256',baselines.catalog()['methods'])
        self.assertNotIn('cellvit256_official',baselines.catalog()['methods'])
    def test_frozen_internal_external_image_roles(self):
        result=baselines.validate_split(self.cfg)
        self.assertEqual(result['sha256'],baselines.catalog()['split_manifest_sha256'])
        for ds,counts in [('consep',(22,5,14)),('crchisto',(68,12,20))]:
            roles=frozen_roles(ds,self.cfg)
            self.assertEqual(tuple(map(len,roles.values())),counts)
            self.assertFalse(set(roles['train']) & set(roles['val']))
            self.assertFalse(set(roles['test']) & (set(roles['train'])|set(roles['val'])))
    def test_edited_frozen_external_split_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'baselines').mkdir()
            data=json.loads((project_root()/'baselines/external_validation_splits.json').read_text())
            data['consep']['train_images'][0],data['consep']['val_images'][0]=data['consep']['val_images'][0],data['consep']['train_images'][0]
            (root/'baselines/external_validation_splits.json').write_text(json.dumps(data))
            with patch('cellnura_release.baseline_runtime.project_root',return_value=root):
                with self.assertRaisesRegex(ValueError,'SHA mismatch'):frozen_roles('consep',self.cfg)
    def test_modified_manifest_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'split.csv';p.write_text(Path(resource('split_manifest',self.cfg)).read_text(encoding='utf-8')+'\n', encoding='utf-8')
            cfg=dict(self.cfg);cfg['baseline_paths']=dict(cfg['baseline_paths'],split_manifest=str(p))
            with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):baselines.validate_split(cfg)
    def test_training_fixed_detector_only_trains_classifier(self):
        cmds=baselines.commands('stardist','train','pannuke',2,self.cfg)
        self.assertEqual(len(cmds),1);self.assertIn('t08_classifier_v3.py',cmds[0][1])
        self.assertNotIn('--eval-all',cmds[0]);self.assertNotIn('--inference-only',cmds[0])
        infer=baselines.commands('stardist','infer','pannuke',2,self.cfg)
        self.assertEqual(len(infer),2);self.assertIn('--inference-only',infer[-1])
    def test_micronet_ranking_uses_retrain_entry(self):
        for action in ('train','infer'):
            cmd=baselines.commands('micronet',action,'pannuke',2,self.cfg)[0]
            self.assertIn('t08_micronet_pannuke_retrain.py',cmd[1])
            self.assertNotIn('t08_micronet_external.py',cmd[1])
        self.assertIn('t08_micronet_consep_retrain.py',baselines.commands('micronet','infer','consep',1,self.cfg)[0][1])
        self.assertIn('t08_micronet_crchisto_retrain.py',baselines.commands('micronet','infer','crchisto',1,self.cfg)[0][1])
    def test_cellnura_cellvit_classifier_uses_fold_training(self):
        cmd=baselines.commands('cellnura_cellvit','train','pannuke',3,self.cfg)[0]
        self.assertIn('baselines/cellnura_cellvit/train_pannuke.py',cmd[1]);self.assertIn('--launch-full',cmd)
        self.assertEqual(cmd[cmd.index('--init')+1],'scratch')
        self.assertEqual(cmd[cmd.index('--outer-fold')+1],'3')
    def test_missing_inference_checkpoint_cannot_start_training(self):
        with tempfile.TemporaryDirectory() as t:
            cfg=dict(self.cfg);cfg['baseline_paths']=dict(cfg['baseline_paths'],runs=str(Path(t)/'missing'))
            with patch.object(baselines.subprocess,'run') as process:
                with self.assertRaises(FileNotFoundError):baselines._inference_checkpoints('dist','pannuke',1,cfg)
                process.assert_not_called()
    def test_weight_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'weight.pth';p.write_bytes(b'expected test weight')
            cfg=dict(self.cfg);cfg['baseline_paths']=dict(cfg['baseline_paths'],pannuke=t,cellvit_sam_weight=str(p))
            cfg['baselines']=dict(cfg['baselines']);cfg['baselines']['cellvit_sam']={'weight_sha256':{'cellvit_sam_weight':hashlib.sha256(p.read_bytes()).hexdigest()}}
            self.assertTrue(baselines.check_method('cellvit_sam',cfg,'pannuke')['passed'])
            p.write_bytes(b'modified test weight')
            result=baselines.check_method('cellvit_sam',cfg,'pannuke')
            self.assertFalse(result['passed']);self.assertTrue(any('Weight SHA mismatch' in e for e in result['errors']))
    def test_unknown_resource_fails(self):
        with self.assertRaises(KeyError):resource('unknown_baseline_data',self.cfg)
    def test_unfetched_tsfd_is_nonzero(self):
        if Path(resource('tsfd_source',self.cfg)).exists():self.skipTest('TSFD explicitly fetched in this installation')
        with contextlib.redirect_stdout(io.StringIO()):status=baselines.main(['check','tsfd','--sources-only'])
        self.assertEqual(status,2)
    def test_sources_parse_and_use_explicit_resources(self):
        for name in baselines.catalog()['methods']:
            if name in ('tsfd','tsfd_public'):continue
            result=baselines.check_method(name,self.cfg,sources_only=True)
            self.assertTrue(result['passed'],result['errors'])

if __name__=='__main__':unittest.main()
