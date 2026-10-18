"""Native-scale HoVer-Net external inference with explicit checkpoint and vocabulary."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from cellnura_release.baseline_runtime import resource,test_image_ids,setting
sys.path.insert(0,resource('vendor_hovernet_original'))
sys.path.insert(0,resource('vendor_hovernet'))
from models.hovernet.net_desc import create_model
from models.hovernet.post_proc import process
import t06_consep_infer as canonical
from t08_hovernet_external_eval import load_gt
from unified_matcher import EvalInput,Instance,evaluate,write_outputs

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',required=True,choices=['consep','crchisto'])
    ap.add_argument('--official',action='store_true')
    a=ap.parse_args();ds=a.dataset
    canonical.NR_TYPES=5
    # Formal CRCHisto independent run selected FP32, avoiding AMP instability.
    if ds=='crchisto' and not a.official:
        class NoAMP:
            def __enter__(self): return self
            def __exit__(self,*args): return False
        canonical.torch.autocast=lambda *args,**kwargs:NoAMP()
    checkpoint=resource('hovernet_official_weight',required=True) if a.official else str(Path(resource('runs'))/f'hovernet_fast_{ds}_v33/best_model.pth')
    mode=str(setting('official_mode','original')) if a.official else 'fast'
    model=create_model(mode=mode,nr_types=5)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    state=ck.get('desc',ck)
    state={k.removeprefix('module.'):v for k,v in state.items()}
    model.load_state_dict(state,strict=True);model=model.cuda().eval()
    if mode not in ('fast','original'): raise ValueError('Unknown HoVer-Net mode')
    canonical.IN,canonical.OUTP=(256,164) if mode=='fast' else (270,80)
    canonical.PAD=(canonical.IN-canonical.OUTP)//2
    gt=load_gt(ds);ids=test_image_ids(ds)
    root=Path(resource(ds))
    dest=Path(resource('evaluation'))/('hovernet_official_' if a.official else 'hovernet_indep_')/ds
    dest.mkdir(parents=True,exist_ok=True);(dest/'detections').mkdir(exist_ok=True)
    all_gt=[];all_pred=[]
    mapping={1:3,2:0,3:1,4:2} if a.official else {1:0,2:1,3:2,4:3}
    for name in ids:
        ip=root/'image'/(name+'.png') if ds=='consep' else root/'CRCHisto_database/Detection'/name/(name+'.bmp')
        image=np.asarray(Image.open(ip).convert('RGB')).astype(np.float32)
        with torch.inference_mode(): pred_map=canonical.infer_image(model,image)
        _,info=process(pred_map,nr_types=5,return_centroids=True)
        records=[]
        for iid,inf in info.items():
            raw=int(inf['type']) if inf['type'] is not None else 0
            cx,cy=map(float,inf['centroid']);cls=mapping.get(raw,-1)
            all_pred.append(Instance(name,str(iid),cx,cy,cls));records.append(dict(instance_id=str(iid),x=cx,y=cy,raw_type=raw,class_id=cls))
        pts,classes=gt[name]
        all_gt.extend(Instance(name,str(i),float(xy[0]),float(xy[1]),int(c)) for i,(xy,c) in enumerate(zip(pts,classes)))
        (dest/'detections'/(name+'.json')).write_text(json.dumps(records))
    ev=EvalInput(ds,5.0 if ds=='consep' else 10.0,all_gt,all_pred,image_ids=ids,real_class_ids=(0,1,2,3))
    write_outputs(evaluate(ev),str(dest/'eval'))
    from dataclasses import asdict
    for kind,values in [('predictions',all_pred),('ground_truth',all_gt)]: (dest/(kind+'.json')).write_text(json.dumps([asdict(v) for v in values]))
    (dest/'run_manifest.json').write_text(json.dumps(dict(method='HoVer-Net',dataset=ds,checkpoint=checkpoint,tier='B' if a.official else 'A',raw_to_unified=mapping,image_ids=ids,adapter='native-scale probability mean windows, canonical t06_consep_infer; not official inference CLI equivalence'),indent=2))
if __name__=='__main__':main()
