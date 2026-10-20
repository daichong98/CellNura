"""Full-GT evaluation for a produced CSV, including zero-detection images."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from cellnura_release.baseline_runtime import resource,test_image_ids,real_classes
from unified_matcher import Instance,EvalInput,evaluate,write_outputs

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',required=True,choices=['pannuke','consep','crchisto'])
    ap.add_argument('--fold',type=int,choices=[1,2,3],default=1)
    ap.add_argument('--predictions',type=Path,required=True)
    ap.add_argument('--output-prefix',type=Path,required=True)
    a=ap.parse_args();ids=test_image_ids(a.dataset,a.fold);gt=[]
    if a.dataset=='pannuke':
        masks=np.load(Path(resource('pannuke'))/f'Fold {a.fold}/masks/fold{a.fold}/masks.npy',mmap_mode='r')
        if set(ids)!={f'fold{a.fold}_image_{i}' for i in range(len(masks))}:raise ValueError('Frozen ids and dataset image count disagree')
        for i,m in enumerate(masks):
            name=f'fold{a.fold}_image_{i}'
            for cls in range(5):
                for iid in np.unique(m[...,cls]):
                    if iid==0:continue
                    yy,xx=np.nonzero(m[...,cls]==iid)
                    gt.append(Instance(name,f'{cls}_{iid}',float(xx.mean()),float(yy.mean()),cls))
    else:
        from t08_hovernet_external_eval import load_gt
        values=load_gt(a.dataset)
        for name in ids:
            xy,classes=values[name]
            gt.extend(Instance(name,str(i),float(c[0]),float(c[1]),int(k)) for i,(c,k) in enumerate(zip(xy,classes)))
    df=pd.read_csv(a.predictions)
    needed={'image_key','nucleus_id','cx','cy','pred'}
    if not needed<=set(df):raise ValueError(f'Prediction CSV missing {needed-set(df)}')
    if not set(df.image_key)<=set(ids):raise ValueError('Prediction image ids outside frozen test set')
    if not np.isfinite(df[['cx','cy','pred']].to_numpy()).all():raise ValueError('Nonfinite predictions')
    pred=[Instance(str(r.image_key),str(r.nucleus_id),float(r.cx),float(r.cy),int(r.pred)) for r in df.itertuples()]
    tau={'pannuke':12.0,'consep':5.0,'crchisto':10.0}[a.dataset]
    result=evaluate(EvalInput(a.dataset,tau,gt,pred,image_ids=ids,real_class_ids=real_classes(a.dataset)))
    a.output_prefix.parent.mkdir(parents=True,exist_ok=True);write_outputs(result,str(a.output_prefix))
    print(f'{a.dataset}: images={len(ids)} N_gt={result.n_gt} N_pred={result.n_pred} unified_real={result.macro("unified"):.8f}')
if __name__=='__main__':main()
