"""Portable baseline catalog, real preflight checks, and explicit process dispatch."""
from __future__ import annotations
import argparse,ast,csv,hashlib,json,os,subprocess,sys
from pathlib import Path
from .paths import project_root,load_config,resolve_path
from .baseline_runtime import resource,frozen_roles

CATALOG=project_root()/'baselines/catalog.json'
_SCRIPT_DIR={
  "t08_train_maskrcnn.py": "maskrcnn",
  "t08_train_dist.py": "dist",
  "t08_train_dist_external.py": "dist",
  "micronet_standalone.py": "micronet",
  "t08_micronet_external.py": "micronet",
  "t08_micronet_pannuke_retrain.py": "micronet",
  "t08_micronet_consep_retrain.py": "micronet",
  "t08_micronet_crchisto_retrain.py": "micronet",
  "t06_train_micronet_consep.py": "micronet",
  "t06_train_micronet_crchisto.py": "micronet",
  "t06_train_micronet_pannuke.py": "micronet",
  "train_hovernet_pannuke_v3.py": "hovernet",
  "t08_hovernet_v3_indep.py": "hovernet",
  "t08_hovernet_tphead_eval.py": "hovernet",
  "t08_hovernet_external_eval.py": "hovernet",
  "t06_v32_infer_per_outer_fold.py": "hovernet",
  "t06_train_hovernet_consep.py": "hovernet",
  "t06_train_hovernet_crchisto.py": "hovernet",
  "t06_consep_infer.py": "hovernet",
  "hovernet_external_infer.py": "hovernet",
  "t08_cellvit_full_pannuke.py": "cellvitpp",
  "t08_cellvitpp_consep.py": "cellvitpp",
  "t09_traditional_classifiers.py": "traditional",
  "baseline_feature_columns.py": "traditional",
  "tsfd_evaluate.py": "tsfd",
  "tsfd_runtime.py": "tsfd",
  "unified_matcher.py": "shared",
  "evaluate_csv.py": "shared",
  "t08_classifier_v3.py": "shared",
  "t08_det_baseline_inference.py": "shared"
}
ENV_MODULES={
 'torch':['numpy','pandas','scipy','cv2','torch','torchvision','skimage','PIL'],
 'cellvit':['numpy','pandas','scipy','torch','torchvision','einops','timm','skimage','cv2','yaml','matplotlib','torchmetrics','albumentations','wandb','tqdm','torchinfo'],
 'cellvitpp':['numpy','pandas','scipy','torch','torchvision','einops','timm','skimage','cv2','yaml','matplotlib'],
 'tsfd':['tensorflow','numpy','pandas','scipy','cv2','skimage','PIL','efficientnet','imgaug'],
 'stardist':['tensorflow','stardist','csbdeep','numpy','pandas','scipy'],
 'traditional':['numpy','pandas','scipy','sklearn'],
}

def catalog(): return json.loads(CATALOG.read_text(encoding='utf-8'))
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()
def _python(cfg,name,classifier=False):
    spec=cfg.get('baselines',{}).get(name,{})
    value=spec.get('classifier_python') if classifier else spec.get('python')
    if value in (None,''):return sys.executable
    path=resolve_path(value,cfg)
    if not path.is_file():raise FileNotFoundError(f'Python interpreter not found: {path}')
    return str(path)
def _env(cfgpath,name,action,dataset,fold):
    env=os.environ.copy();env['CELLNURA_CONFIG_PATH']=str(cfgpath)
    env['CELLNURA_BASELINE_NAME']=name;env['CELLNURA_BASELINE_ACTION']=action
    env['CELLNURA_BASELINE_DATASET']=dataset;env['CELLNURA_BASELINE_FOLD']=str(fold)
    base=project_root()/'baselines'
    folders=('shared','maskrcnn','dist','micronet','hovernet','cellvitpp','traditional','tsfd','cellnura_cellvit')
    env['PYTHONPATH']=os.pathsep.join([str(project_root()),*(str(base/name) for name in folders),env.get('PYTHONPATH','')])
    return env

def validate_split(cfg):
    path=resource('split_manifest',cfg,required=True)
    expected=catalog()['split_manifest_sha256'];actual=sha256(path)
    if actual!=expected:raise ValueError(f'Frozen split SHA-256 mismatch: {actual} != {expected}')
    with open(path,encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    groups={}
    for r in rows:
        key=(r['dataset'],r['outer_fold']);groups.setdefault(key,{'train':set(),'val':set(),'test':set()})
        if r['split'] not in groups[key]:raise ValueError(f'Unknown split role: {r["split"]}')
        if r['image_id'] in groups[key][r['split']]:raise ValueError(f'Duplicate image row in split: {key}/{r["image_id"]}')
        groups[key][r['split']].add(r['image_id'])
    for key,g in groups.items():
        required=('train','val','test') if key[0]=='PanNuke' else ('train','test')
        if any(not g[k] for k in required):raise ValueError(f'Incomplete split group {key}')
        if (g['train']&g['val']) or (g['train']&g['test']) or (g['val']&g['test']):raise ValueError(f'Image split leakage: {key}')
    for ds in ('consep','crchisto'): frozen_roles(ds,cfg)
    return {'path':str(path),'sha256':actual,'groups':{str(k):{role:len(ids) for role,ids in g.items()} for k,g in groups.items()}}

def _source_path(name):return project_root()/'baselines'/name

def check_method(name,cfg,dataset=None,action='infer',sources_only=False,environment=False):
    spec=catalog()['methods'][name];errors=[];checks=[]
    for f in spec['sources']:
        p=_source_path(f)
        try:
            s=p.read_text(encoding='utf-8');ast.parse(s,filename=str(p))
            if '/Data/' in s or '/Other/' in s:raise ValueError('Workstation path in portable script')
            expected=spec.get('source_sha256',{}).get(f)
            if expected and sha256(p)!=expected:raise ValueError('Portable source SHA mismatch; regenerate release manifest after an intentional source change')
            checks.append({'source':str(p.relative_to(project_root())),'sha256':sha256(p),'passed':True})
        except Exception as exc:errors.append(f'{f}: {exc}')
    for vendor in spec['vendor']:
        if vendor=='tsfd':
            root=Path(resource('tsfd_source',cfg))
            missing=[f for f in ('scripts/models.py','scripts/bifpn.py','scripts/losses.py','slide_inference/utils.py') if not(root/f).is_file()]
            manifest=root/'FETCH_MANIFEST.json'
            if missing:errors.append('TSFD upstream not fetched: '+', '.join(missing)+'; run python third_party/fetch_source.py tsfd')
            elif not manifest.is_file():errors.append('TSFD source lacks pinned FETCH_MANIFEST.json; use the fetch tool')
            else:
                m=json.loads(manifest.read_text())
                if m.get('commit')!='7620aaf02b9367f898a8d8f1a097fa508e0c55d4':errors.append('TSFD source commit mismatch')
                for f,digest in m.get('files',{}).items():
                    if not(root/f).is_file() or sha256(root/f)!=digest:errors.append('TSFD source SHA mismatch: '+f)
            continue
        root=project_root()/'third_party'/vendor
        manifest=root/'SNAPSHOT.json'
        if manifest.exists():
            for f,digest in json.loads(manifest.read_text())['files'].items():
                if not(root/f).is_file() or sha256(root/f)!=digest:errors.append(f'{vendor} vendored SHA mismatch: {f}')
        elif not(root/'LICENSE.upstream').exists():errors.append('Missing vendored source/license: '+vendor)
    if not sources_only:
        ds=dataset or spec['datasets'][0]
        try: checks.append({'frozen_split':validate_split(cfg)})
        except Exception as exc:errors.append(str(exc))
        try: resource(ds,cfg,required=True)
        except Exception as exc:errors.append(str(exc))
        keys=spec['weights_train' if action=='train' else 'weights_infer']
        for key in keys:
            try:
                p=Path(resource(key,cfg,required=True));expected=cfg.get('baselines',{}).get(name,{}).get('weight_sha256',{}).get(key)
                if key.endswith('_dir'):
                    if not p.is_dir():raise ValueError(f'{key} must be a directory')
                    if not isinstance(expected,dict) or not expected:raise ValueError(f'Configure baselines.{name}.weight_sha256.{key} as a file→SHA map')
                    for f,digest in expected.items():
                        if sha256(p/f)!=digest:raise ValueError(f'Weight SHA mismatch: {key}/{f}')
                else:
                    if not expected or len(expected)!=64:raise ValueError(f'Configure baselines.{name}.weight_sha256.{key} with verified SHA-256')
                    if sha256(p)!=expected:raise ValueError(f'Weight SHA mismatch: {key}')
                checks.append({'weight':key,'sha256_verified':True})
            except Exception as exc:errors.append(str(exc))
        if name=='traditional':
            try:resource('traditional_features',cfg,required=True)
            except Exception as exc:errors.append(str(exc))
    if environment:
        profiles=[(spec['environment'],False)]
        if name in ('stardist','micronet','dist'):profiles.append(('torch',True))
        for profile,classifier in profiles:
            try:
                modules=ENV_MODULES[profile]
                env=_env(cfg['_config_path'],name,action,dataset or spec['datasets'][0],1)
                code='import importlib,json; names='+repr(modules)+'; result={};\nfor name in names:\n try:\n  m=importlib.import_module(name); result[name]="OK "+str(getattr(m,"__version__",""))\n except Exception as e: result[name]="ERROR "+repr(e)\n'
                if profile not in ('traditional','stardist'):
                    if profile=='tsfd':
                        code+='\ntry:\n import tensorflow as tf; gpu=bool(tf.config.list_physical_devices("GPU"))\nexcept Exception: gpu=False\n'
                    else:
                        code+='\ntry:\n import torch; gpu=torch.cuda.is_available()\nexcept Exception: gpu=False\n'
                    code+='result["GPU"]="OK CUDA" if gpu else "ERROR GPU unavailable; this adapter requires CUDA"\n'
                code+='print(json.dumps(result)); raise SystemExit(any(v.startswith("ERROR") for v in result.values()))'
                p=subprocess.run([_python(cfg,name,classifier),'-c',code],env=env,cwd=project_root(),capture_output=True,text=True,timeout=90)
                checks.append({'environment_profile':profile,'classifier':classifier,'environment':p.stdout.strip()})
                if p.returncode:errors.append(('Classifier ' if classifier else '')+'environment imports/GPU failed: '+p.stdout[-1200:]+' '+p.stderr[-500:])
            except Exception as exc:errors.append('Environment check failed: '+str(exc))
    return {'method':name,'passed':not errors,'scope':'source integrity only; no data/weight/runtime claim' if sources_only else 'source+frozen image split+configured input weights','checks':checks,'errors':errors}

def commands(name,action,ds,fold,cfg,extra=()):
    """Return concrete argv pipelines. Extra args target the first method-specific process."""
    py=_python(cfg,name);cpy=_python(cfg,name,True);B=project_root()/'baselines';T=B/'tsfd'
    run=Path(resource('runs',cfg));det=Path(resource('detections',cfg));ev=Path(resource('evaluation',cfg))
    def cmd(script,*args,classifier=False):return [cpy if classifier else py,str(B/_SCRIPT_DIR[script]/script),*map(str,args)]
    shared=lambda infer,eval_all=True:cmd('t08_classifier_v3.py','--test-fold',fold,*(['--eval-all'] if eval_all else []),'--method',{'micronet':'Micro-Net','stardist':'StarDist','dist':'DIST'}[name],*(['--inference-only'] if infer else []),classifier=True)
    if name=='maskrcnn':
        items=[cmd('t08_train_maskrcnn.py','--dataset',ds,'--test-fold',fold,'--train' if action=='train' else '--infer',*(['--epochs',30 if ds=='pannuke' else 60] if action=='train' else []))]
        if action=='infer':items.append(cmd('evaluate_csv.py','--dataset',ds,'--fold',fold,'--predictions',run/f'maskrcnn_{ds}_testfold{fold}/pred_all.csv','--output-prefix',ev/f'maskrcnn_{ds}/eval_fold{fold}'))
    elif name=='dist':
        if ds=='pannuke':items=[cmd('t08_train_dist.py','--test-fold',fold,*(['--infer-only'] if action=='infer' else [])),shared(action=='infer')]
        else:items=[cmd('t08_train_dist_external.py','--dataset',ds)]
    elif name=='stardist':
        if ds!='pannuke':raise ValueError('StarDist ranking runs are PanNuke only')
        items=[shared(False,eval_all=False)] if action=='train' else [cmd('t08_det_baseline_inference.py','--method',name,'--folds',fold,'--output-dir',det/f'{name}_full'),shared(True)]
    elif name=='micronet':
        # Ranking rows are within-split Micro-Net retraining, not the fixed TIAToolbox detector.
        if action=='train':
            if ds=='pannuke':items=[cmd('t06_train_micronet_pannuke.py','--folds',fold)]
            elif ds=='consep':items=[cmd('t06_train_micronet_consep.py')]
            elif ds=='crchisto':items=[cmd('t06_train_micronet_crchisto.py')]
            else:raise ValueError('Micro-Net supports pannuke, consep, and crchisto')
        elif ds=='pannuke':items=[cmd('t08_micronet_pannuke_retrain.py','--folds',fold)]
        elif ds=='consep':items=[cmd('t08_micronet_consep_retrain.py')]
        elif ds=='crchisto':items=[cmd('t08_micronet_crchisto_retrain.py')]
        else:raise ValueError('Micro-Net supports pannuke, consep, and crchisto')
    elif name=='hovernet':
        if ds=='pannuke':items=[cmd('train_hovernet_pannuke_v3.py','--test-fold',fold)] if action=='train' else [cmd('t06_v32_infer_per_outer_fold.py','--outer-fold',fold,'--test-only'),cmd('t08_hovernet_v3_indep.py')]
        else:items=[cmd('t06_train_hovernet_'+ds+'.py')] if action=='train' else [cmd('hovernet_external_infer.py','--dataset',ds)]
    elif name=='cellnura_cellvit':
        if ds=='pannuke':
            if action=='train':items=[[py,(project_root()/'baselines/cellnura_cellvit/train_pannuke.py').as_posix(),'--outer-fold',str(fold),'--init','scratch','--launch-full','--output-root',str(run/'cellnura_cellvit')]]
        elif action=='train':items=[[py,(project_root()/'baselines/cellnura_cellvit/train_external.py').as_posix(),'--dataset',ds]]
        if action=='infer':
            spec=cfg.get('baselines',{}).get('cellnura_cellvit',{})
            base=spec.get('frozen_detections',{}).get(ds,{})
            base=base.get(str(fold)) if ds=='pannuke' and isinstance(base,dict) else base
            if not base:raise ValueError('Configure baselines.cellnura_cellvit.frozen_detections for this dataset/fold')
            base=resolve_path(base,cfg)
            images=project_root()/'artifacts/images'/(f'pannuke_fold{fold}.csv' if ds=='pannuke' else f'{ds}.csv')
            ckpt,digest=_cellnura_cellvit_checkpoint(cfg,ds,fold)
            dest=ev/'cellnura_cellvit'/(f'pannuke_fold{fold}.csv' if ds=='pannuke' else f'{ds}.csv')
            items=[[py,(project_root()/'baselines/cellnura_cellvit/infer_fixed_detections.py').as_posix(),
                    '--dataset',ds,'--fold',str(fold),'--dataset-root',str(resource(ds,cfg)),
                    '--image-list',str(images),'--frozen-detections',str(base),
                    '--checkpoint',str(ckpt),'--checkpoint-sha256',digest,'--output',str(dest)]]
    elif name=='tsfd':
        folder=run/'tsfd'/f'fold{fold}'
        if action=='train':items=[[py,str(T/'prepare_cache.py')],[py,str(T/'train.py'),'--fold',str(fold),'--output-dir',str(folder)]]
        else:items=[[py,str(T/'evaluate_fold.py'),'--run-dir',str(folder),'--output-dir',str(ev/'tsfd'/f'fold{fold}')]]
    elif name=='cellvit_sam':
        if ds!='pannuke':raise ValueError('CellViT-SAM external CoNSeP/CRCHisto results are outside the manuscript and supplementary tables')
        items=[cmd('t08_cellvit_full_pannuke.py')]
    elif name=='cellvitpp':items=[cmd('t08_cellvitpp_consep.py','--output-dir',ev/'cellvitpp_consep')]
    elif name=='hovernet_official':items=[cmd('hovernet_external_infer.py','--dataset',ds,'--official')]
    elif name=='hovernet_tphead':items=[cmd('t08_hovernet_tphead_eval.py')]
    elif name=='traditional':
        models=['SVM_RBF','SVM_Linear','RandomForest','GradientBoosting','LogisticRegression','NaiveBayes','KNN','MLP_Sklearn']
        selected=cfg.get('baselines',{}).get(name,{}).get('models',models)
        if not selected or any(n not in models for n in selected):raise ValueError('Unknown/empty traditional models')
        items=[cmd('t09_traditional_classifiers.py','--model',m,'--test-fold',fold) for m in selected]
    else:raise ValueError('Unknown baseline '+name)
    if extra:items[0].extend(extra)
    return items

def _cellnura_cellvit_checkpoint(cfg,ds,fold):
    spec=cfg.get('baselines',{}).get('cellnura_cellvit',{})
    key=f'pannuke_fold{fold}' if ds=='pannuke' else ds
    explicit=spec.get('checkpoints',{}).get(key)
    expected=spec.get('checkpoint_sha256',{}).get(key)
    if explicit:
        path=resolve_path(explicit,cfg)
        if not expected or sha256(path)!=expected:raise ValueError('CellViT same-detection classifier checkpoint SHA mismatch/missing')
        return path,expected
    root=Path(resource('runs',cfg))/'cellnura_cellvit'
    if ds=='pannuke':
        report=root/f'outer_fold{fold}'/'run_manifest.json'
        if not report.is_file():raise FileNotFoundError('CellViT same-detection classifier trainer manifest missing; train first or configure checkpoint')
        m=json.loads(report.read_text())
        if m.get('status')!='training_finished' or m.get('initialization')!='scratch':raise ValueError('CellViT same-detection classifier manifest is not completed random-initialized training')
        path=root/m['checkpoint_relative']
        expected=m['checkpoint_sha256']
    else:
        path=Path(resource('runs',cfg))/f'cellnura_cellvit_classifier_{ds}'/'best_model.pth'
        if not path.is_file():raise FileNotFoundError('CellViT same-detection external classifier checkpoint missing; train first or configure checkpoint')
        report=path.with_name('run_manifest.json')
        if not report.is_file():raise FileNotFoundError('CellViT same-detection classifier run manifest missing')
        m=json.loads(report.read_text())
        if not expected:expected=sha256(path)
    if sha256(path)!=expected:raise ValueError('CellViT same-detection classifier checkpoint SHA mismatch')
    return path,expected

def _verify_distance_checkpoint(path,cfg):
    code="import sys,torch;sys.path.insert(0,sys.argv[1]);from model import DIST;m=DIST();s=torch.load(sys.argv[2],map_location='cpu',weights_only=True);m.load_state_dict(s,strict=True);print('DIST_DISTANCE_ARCHITECTURE_OK')"
    result=subprocess.run([_python(cfg,'dist'),'-c',code,resource('vendor_dist_port',cfg),str(path)],capture_output=True,text=True,cwd=project_root())
    if result.returncode:raise ValueError('Distance checkpoint is not compatible with DIST; historical external best_model.pth is a ResNet classifier and cannot be used as the detector. Restore the independent distance checkpoint or retrain. '+result.stderr[-500:])

def _inference_checkpoints(name,ds,fold,cfg):
    r=Path(resource('runs',cfg))
    if name in ('dist','stardist') and ds=='pannuke':
        paths=[r/f'resnet18_v3_classifier_testfold{fold}/best_model.pth',r/f'resnet18_v3_classifier_testfold{fold}/run_manifest.json']
        if name=='dist':paths.append(r/f'dist_pannuke_testfold{fold}/best_model.pth')
    elif name=='maskrcnn':paths=[r/f'maskrcnn_{ds}_testfold{fold}/best_model.pth']
    elif name=='hovernet':paths=[r/(f'hovernet_v3_fast_pannuke_testfold{fold}' if ds=='pannuke' else f'hovernet_fast_{ds}_v33')/'best_model.pth']
    elif name=='cellnura_cellvit':paths=[_cellnura_cellvit_checkpoint(cfg,ds,fold)[0]]
    elif name=='tsfd':paths=[r/'tsfd'/f'fold{fold}'/'best_model.h5',r/'tsfd'/f'fold{fold}'/'TRAINING_MANIFEST.json']
    elif name=='dist':paths=[r/f'dist_scratch_{ds}/best_model.pth',r/f'dist_scratch_{ds}/classifier/best_model.pth']
    elif name=='micronet':
        if ds=='pannuke':paths=[r/f'micronet_retrain_pannuke_fold{fold}/best_model.pth']
        elif ds=='consep':paths=[r/'micronet_retrain_consep'/'best_model.pth']
        else:paths=[r/'micronet_retrain_crchisto'/'best_model.pth']
    elif name=='hovernet_tphead':paths=[Path(resource('tphead_runs',cfg))/f'cv{fold}_tp_head.pth',Path(resource('official_hovernet_detections',cfg))]
    else:paths=[]
    for path in paths:
        if not path.exists():raise FileNotFoundError(f'Trained input missing; run train or restore externally supplied checkpoint tree: {path}')
    if name=='dist':_verify_distance_checkpoint(r/(f'dist_pannuke_testfold{fold}' if ds=='pannuke' else f'dist_scratch_{ds}')/'best_model.pth',cfg)
    return paths

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    ls=sub.add_parser('list',help='show exact implementations, eligibility and available actions');ls.add_argument('--json',action='store_true')
    ck=sub.add_parser('check',help='verify source, configured data, split, input weight hashes and optional environment')
    ck.add_argument('name',choices=['all',*catalog()['methods']]);ck.add_argument('--config');ck.add_argument('--sources-only',action='store_true');ck.add_argument('--environment',action='store_true');ck.add_argument('--dataset',choices=['pannuke','consep','crchisto']);ck.add_argument('--action',choices=['train','infer'],default='infer');ck.add_argument('--json',action='store_true')
    run=sub.add_parser('run',help='run an actual training/inference pipeline, never download weights implicitly')
    run.add_argument('name',choices=list(catalog()['methods']));run.add_argument('action',choices=['train','infer']);run.add_argument('--config');run.add_argument('--dataset',choices=['pannuke','consep','crchisto'],default='pannuke');run.add_argument('--fold',type=int,choices=[1,2,3],default=1);run.add_argument('--dry-run',action='store_true',help='print argv only; explicitly not a successful execution')
    args,extra=parser.parse_known_args(argv)
    if extra and args.command!='run':parser.error('Unrecognized arguments: '+' '.join(extra))
    if extra and extra[0]=='--':extra=extra[1:]
    if args.command=='list':
        if args.json:print(json.dumps(catalog(),ensure_ascii=False,indent=2))
        else:
            for name,spec in catalog()['methods'].items():print(f'{name:20} {",".join(spec["actions"]):12} {spec["tier"]} {spec["title"]}')
        return 0
    try:
        cfg=load_config(args.config)
        if args.command=='check':
            names=list(catalog()['methods']) if args.name=='all' else [args.name]
            results=[check_method(n,cfg,args.dataset,args.action,args.sources_only,args.environment) for n in names]
            print(json.dumps(results,ensure_ascii=False,indent=2) if args.json else '\n'.join(f'{v["method"]}: {"PASS" if v["passed"] else "FAIL"} — {v["scope"]}\n'+ '\n'.join('  '+e for e in v['errors']) for v in results))
            return 0 if all(r['passed'] for r in results) else 2
        spec=catalog()['methods'][args.name]
        if args.action not in spec['actions']:raise ValueError(f'{args.name} supports {spec["actions"]}, not {args.action}')
        if args.dataset not in spec['datasets']:raise ValueError(f'{args.name} supports datasets {spec["datasets"]}, not {args.dataset}')
        cmds=commands(args.name,args.action,args.dataset,args.fold,cfg,extra)
        if args.dry_run:
            print(json.dumps({'executed':False,'dry_run':True,'commands':cmds},ensure_ascii=False,indent=2));return 0
        result=check_method(args.name,cfg,args.dataset,args.action,environment=True)
        if not result['passed']:raise ValueError('\n'.join(result['errors']))
        if args.action=='infer':_inference_checkpoints(args.name,args.dataset,args.fold,cfg)
        env=_env(cfg['_config_path'],args.name,args.action,args.dataset,args.fold)
        root=Path(resource('runs',cfg));root.mkdir(parents=True,exist_ok=True)
        receipt=root/f'RELEASE_{args.name}_{args.dataset}_fold{args.fold}_{args.action}.json'
        receipt.write_text(json.dumps({'completed':False,'config_sha256':sha256(cfg['_config_path']),'commands':cmds,'eligibility':spec['tier'][args.dataset]},ensure_ascii=False,indent=2))
        for cmd in cmds:subprocess.run(cmd,check=True,cwd=project_root(),env=env)
        data=json.loads(receipt.read_text());data['completed']=True;receipt.write_text(json.dumps(data,ensure_ascii=False,indent=2))
        return 0
    except (ValueError,KeyError,FileNotFoundError,subprocess.CalledProcessError) as exc:
        print(f'Baseline error: {exc}',file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
