#!/usr/bin/env python3
"""Evaluate only a completed independently trained fold, using the existing unified matcher."""
import argparse,sys,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
from common import *
sys.path.insert(0,str(CODE))
from tsfd_runtime import initialize,prepare,finish
import tsfd_evaluate
def proof(run):
    run=Path(run)
    m=json.loads((run/"TRAINING_MANIFEST.json").read_text())
    c=json.loads((run/"training_configuration.json").read_text())
    assert m["status"]=="training_complete" and m["completed_epochs"]==c["epochs"]==150
    assert m["configuration_sha256"]==sha(run/"training_configuration.json")
    assert m["independent_initialization"] and m["train_val_test_disjoint"]
    assert not m["test_used_for_training"] and not m["test_used_for_selection"]
    assert c["checkpoint_initialization_path"] is None
    assert c["initialization"]=="random; author task weights and ef4.h5 are never loaded"
    sp=splits(c["outer_fold"])
    assert c["split_sha256"]==SPLIT_SHA
    assert c["train_image_ids"]==sp["train"] and c["val_image_ids"]==sp["val"]
    assert c["test_image_ids_reserved"]==sp["test"]
    checkpoint=run/"best_model.h5"
    assert sha(checkpoint)==m["best_model_sha256"]
    return dict(fold=c["outer_fold"],checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        training_configuration_sha256=sha(run/"training_configuration.json"),
        training_manifest_sha256=sha(run/"TRAINING_MANIFEST.json"),split_sha256=SPLIT_SHA,
        independent_initialization=True,train_val_test_disjoint=True,test_used_for_training=False,
        test_used_for_selection=False,best_epoch=m["best_epoch"],selection=c["selection"],
        protocol="TSFD architecture, corrected author losses, independently retrained from scratch on frozen splits")
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",type=Path,required=True);ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--batch-size",type=int,default=32);ap.add_argument("--workers",type=int,default=16)
    a=ap.parse_args();p=proof(a.run_dir);fold=p["fold"]
    out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    keys=[key(n) for n in splits(fold)["test"]]
    config=dict(keys=keys,checkpoint_sha256=p["checkpoint_sha256"],split_sha256=SPLIT_SHA,
                batch_size=a.batch_size,workers=a.workers,training_provenance=p,
                postprocessing_source_sha256=sha(CODE/"tsfd_runtime.py"),
                inference_code_sha256=sha(Path(__file__)),
                author_postprocessing_sha256=sha(Path(resource("tsfd_source"))/"slide_inference/utils.py"))
    if (out/"configuration.json").exists():
        assert json.loads((out/"configuration.json").read_text())==json.loads(json.dumps(config))
    else:write_json(out/"configuration.json",config)
    for s in ("records","instances","raw"):(out/f"fold{fold}"/s).mkdir(parents=True,exist_ok=True)
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.experimental.enable_op_determinism()
    gpus=tf.config.list_physical_devices("GPU")
    if not gpus:raise RuntimeError("GPU unavailable")
    for g in gpus:tf.config.experimental.set_memory_growth(g,True)
    model=tf.keras.models.load_model(p["checkpoint"],compile=False)
    @tf.function(input_signature=[tf.TensorSpec([None,512,512,6],tf.float32)])
    def infer(x):
        _,seg,bdr=model(x,training=False)
        return tf.cast(seg,tf.float32),tf.cast(bdr,tf.float32)
    remaining=[];done_count=0
    for f,i in keys:
        path=out/f"fold{f}/records/fold{f}_image_{i}.json"
        if path.exists():
            assert json.loads(path.read_text())["image_id"]==f"fold{f}_image_{i}"
            assert (out/f"fold{f}/instances/fold{f}_image_{i}.npz").exists()
            done_count+=1
        else:remaining.append((f,i))
    tick=time.time()
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context("spawn"),initializer=initialize) as pool:
        pending=set();prepared={};submitted=0;window=max(a.batch_size*2,a.workers*2)
        for start in range(0,len(remaining),a.batch_size):
            while submitted<min(len(remaining),start+window):
                k=remaining[submitted];prepared[k]=pool.submit(prepare,k);submitted+=1
            batchkeys=remaining[start:start+a.batch_size]
            x=np.stack([prepared.pop(k).result() for k in batchkeys]).astype(np.float32)
            seg,bdr=infer(x);seg,bdr=seg.numpy(),bdr.numpy()
            assert np.isfinite(seg).all() and np.isfinite(bdr).all()
            for j,(f,i) in enumerate(batchkeys):
                if i<8:np.savez_compressed(out/f"fold{f}/raw/fold{f}_image_{i}.npz",seg=seg[j],boundary=bdr[j])
                pending.add(pool.submit(finish,(f,i,seg[j],bdr[j],str(out))))
            ready={v for v in pending if v.done()}
            for v in ready:v.result();done_count+=1
            pending-=ready
            while len(pending)>window*2:
                ready,pending=wait(pending,return_when=FIRST_COMPLETED)
                for v in ready:v.result();done_count+=1
            if (start//a.batch_size)%10==0:
                state=dict(status="inference",fold=fold,completed=done_count,total=len(keys),seconds=time.time()-tick)
                write_json(out/"progress.json",state);print(json.dumps(state),flush=True)
        for v in pending:v.result();done_count+=1
    assert done_count==len(keys)
    result=tsfd_evaluate.run(out,[fold])
    assert result["passed"] and result["images"]==len(keys)
    result.update(paper_main_table_eligible=True,
                  limitation="Report as TSFD-Net independently retrained from scratch under the frozen protocol.",
                  training_provenance=p)
    write_json(out/"ACCEPTANCE.json",result)
    write_json(out/"run_manifest.json",dict(status="complete",images=len(keys),seconds=time.time()-tick,
        precision="float32",tau=12,
        classes=tsfd_evaluate.NAMES,training_provenance=p,paper_main_table_eligible=True))
    write_json(out/"progress.json",dict(status="complete",fold=fold,images=len(keys)))
    print("FOLD EVALUATION COMPLETE",fold,flush=True)
if __name__=="__main__":main()
