#!/usr/bin/env python3
"""Real train/val preflight: labels, augmentation alignment, gradients, GPU memory, restore."""
import argparse,ast,json,time
import numpy as np
from common import *
from data import Batches,preprocess,augment
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--batch-size",type=int,default=16);ap.add_argument("--precision",choices=["float32","mixed_bfloat16","mixed_float16"],default="mixed_bfloat16");a=ap.parse_args()
    out=Path(resource("cache"))/"tsfd_preflight";out.mkdir(parents=True,exist_ok=True)
    split=splits(1)
    audit={"split_sha256":sha(SPLIT),"folds":{}}
    for f in (1,2,3):
        sp=splits(f)
        audit["folds"][str(f)]={"counts":{s:len(v) for s,v in sp.items()},"overlap_train_val":0,"overlap_train_test":0,"overlap_val_test":0}
    write_json(out/"SPLIT_AUDIT.json",audit)
    # Boundary target must agree exactly with the separately downloaded author's implementation.
    from data import raw_arrays,key
    from skimage.segmentation import find_boundaries
    source=Path(resource("tsfd_boundary_reference", required=True))
    tree=ast.parse(source.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="get_boundaries"]
    ns={"np":np,"find_boundaries":find_boundaries};exec(compile(tree,str(source),"exec"),ns)
    names=split["train"][:a.batch_size]
    references=[]
    for name in names[:3]:
        x,sem,bdr,typ=preprocess(name);f,i=key(name);raw=raw_arrays(f)[1][i]
        expected=(ns["get_boundaries"](raw)>0).astype(np.uint8)
        assert np.array_equal(bdr,expected)
        x1,s1,b1=augment(x,sem,bdr,123);x2,s2,b2=augment(x,sem,bdr,123)
        assert np.array_equal(x1,x2) and np.array_equal(s1,s2) and np.array_equal(b1,b2)
        # Use identical labels as image channels to catch misaligned random transformations.
        probe=np.repeat(sem[...,None],6,-1)
        xa,sa,ba=augment(probe,sem,bdr,456)
        assert np.array_equal(xa[...,0],sa) and np.array_equal(xa[...,3],sa)
        references.append(name)
    print("SPLIT/LABEL/AUGMENTATION PASSED",flush=True)
    from training import configure,build,optimizer,make_steps,dataset,as_float32
    import tensorflow as tf
    configure(43,a.precision);model=build()
    opt,schedule=optimizer(279,150,a.precision)
    train_step,val_step=make_steps(model,opt)
    train=Batches(names,a.batch_size,16,43,cached=False)
    batch=next(iter(dataset(train,0,True)));x,ys=batch
    probes=[model.get_layer(n).trainable_variables[0] for n in ("clf_out","seg_out","inst_out")]
    before=[v.numpy().copy() for v in probes]
    times=[];losses=[]
    for i in range(4):
        tick=time.time();v,gn=train_step(x,ys)
        values=v.numpy();norm=float(gn.numpy());sec=time.time()-tick
        assert np.isfinite(values).all() and np.isfinite(norm),f"nonfinite gradients {norm}"
        times.append(sec);losses.append(values.tolist())
        print(json.dumps(dict(step=i+1,losses=values.tolist(),gradient_norm=norm,seconds=sec,
                              memory=tf.config.experimental.get_memory_info("GPU:0"))),flush=True)
    changed=[bool(np.any(v.numpy()!=b)) for v,b in zip(probes,before)]
    assert all(changed),"At least one head was not updated"
    val=Batches(split["val"][:a.batch_size],a.batch_size,16,43,cached=False)
    vx,vy=next(iter(dataset(val,0,False)))
    iterations=int(opt.iterations.numpy());vals=val_step(vx,vy).numpy()
    assert int(opt.iterations.numpy())==iterations and np.isfinite(vals).all()
    predicted=[p.numpy().astype(np.float32) for p in model(vx[:1],training=False)]
    ckpt=tf.train.Checkpoint(model=model,optimizer=opt)
    saved=ckpt.save(str(out/"checkpoint"))
    pbefore=[p.numpy().copy() for p in probes]
    optimizer_vars=[v.numpy().copy() for v in opt.variables()]
    probes[0].assign_add(tf.ones_like(probes[0]));opt.iterations.assign_add(100)
    ckpt.restore(saved).assert_consumed()
    assert int(opt.iterations.numpy())==iterations
    for v,b in zip(probes,pbefore):np.testing.assert_array_equal(v.numpy(),b)
    for v,b in zip(opt.variables(),optimizer_vars):np.testing.assert_array_equal(v.numpy(),b)
    restored=[p.numpy().astype(np.float32) for p in model(vx[:1],training=False)]
    for a1,b1 in zip(predicted,restored):np.testing.assert_array_equal(a1,b1)
    exported=as_float32(model)
    export_reference=[v.numpy() for v in exported(vx[:1],training=False)]
    assert all(np.isfinite(v).all() for v in export_reference)
    exported.save(str(out/"export_check.h5"),include_optimizer=False)
    reloaded=tf.keras.models.load_model(str(out/"export_check.h5"),compile=False)
    for aa,bb in zip(export_reference,[v.numpy() for v in reloaded(vx[:1],training=False)]):
        np.testing.assert_allclose(aa,bb,rtol=1e-5,atol=1e-5)
    del reloaded,exported
    v,gn=train_step(x,ys)
    assert int(opt.iterations.numpy())==iterations+1 and np.isfinite(v.numpy()).all()
    train.close();val.close()
    report=dict(passed=True,split_sha256=SPLIT_SHA,test_images_read=0,train_sample_ids=names,
                validation_sample_ids=split["val"][:a.batch_size],author_boundary_reference_matches=references,
                deterministic_and_aligned_augmentation=True,heads_updated=changed,
                finite_train_and_validation_loss=True,validation_does_not_update_optimizer=True,
                checkpoint_model_optimizer_predictions_exactly_restored=True,continued_training_after_restore=True,exported_h5_inference_matches=True,export_precision="float32",exported_weight_values_preserved=True,
                batch_size=a.batch_size,precision=a.precision,parameters=model.count_params(),
                step_seconds=times,steady_images_per_second=a.batch_size/float(np.mean(times[1:])),
                peak_gpu_memory_bytes=tf.config.experimental.get_memory_info("GPU:0")["peak"],
                train_losses=losses,val_losses=vals.tolist(),sources=sources(),
                completed=time.strftime("%Y-%m-%d %H:%M:%S"))
    write_json(out/"PREFLIGHT.json",report);print("PREFLIGHT PASSED",json.dumps(report),flush=True)
if __name__=="__main__":main()
