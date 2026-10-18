#!/usr/bin/env python3
"""Independent TSFD training: frozen train/val only; test is a separate post-training stage."""
import argparse,os,time,math,csv,traceback
import numpy as np
from common import *
from data import Batches
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fold",type=int,choices=[1,2,3],required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--epochs",type=int,default=150)
    ap.add_argument("--batch-size",type=int,default=16)
    ap.add_argument("--workers",type=int,default=16)
    ap.add_argument("--precision",choices=["mixed_bfloat16","mixed_float16","float32"],default="mixed_bfloat16")
    ap.add_argument("--resume",action="store_true")
    a=ap.parse_args()
    if a.epochs<=3:ap.error("epochs must exceed warmup=3")
    output=a.output_dir.resolve();output.mkdir(parents=True,exist_ok=a.resume)
    split=splits(a.fold);seed=42+a.fold;steps=math.ceil(len(split["train"])/a.batch_size)
    cache=json.loads((CACHE/"COMPLETE.json").read_text())
    assert cache["signature"]["data_code_sha256"]==sha(HERE/"data.py")
    config=dict(method="TSFD-Net",variant="fold-specific scratch retraining",
        outer_fold=a.fold,seed=seed,epochs=a.epochs,batch_size=a.batch_size,cpu_workers=a.workers,
        precision=a.precision,inference_precision="float32",initialization="random; author task weights and ef4.h5 are never loaded",
        checkpoint_initialization_path=None,split_sha256=SPLIT_SHA,
        split_counts={s:len(ids) for s,ids in split.items()},train_image_ids=split["train"],val_image_ids=split["val"],
        test_image_ids_reserved=split["test"],steps_per_epoch=steps,
        optimizer={"name":"SGD","momentum":.9,"weight_decay":5e-4,"peak_lr":.16,"warmup_epochs":3,"decay":"cosine"},
        model={"bifpn_channels":160,"bifpn_repeats":3,"dropout":.2,"input":[512,512,6],"outputs":[19,6,1]},
        loss={"clf":"author FocalLoss","seg":"author FocalTverskyLoss + 0.4*Weighted_BCEnDice_loss",
              "boundary":"author FocalTverskyLoss + 0.4*Combo_loss","branch_weights":[1,1,1],"regularization":"author model L2 retained"},
        selection="minimum full validation total loss; no test-based selection",
        postprocessing="fixed author threshold=0.5,min_area=22,kernel=3, no test tuning",
        source_sha256=sources(),
        resume_semantics="model, optimizer momentum/iterations and epoch restored; deterministic sample augmentation; dropout RNG stream may differ after process restart")
    configpath=output/"training_configuration.json"
    if configpath.exists():assert json.loads(configpath.read_text())==config,"Resume configuration changed"
    else:write_json(configpath,config)
    from training import configure,build,optimizer,make_steps,dataset,as_float32
    import tensorflow as tf
    configure(seed,a.precision)
    model=build()
    opt,schedule=optimizer(steps,a.epochs,a.precision)
    train_step,val_step=make_steps(model,opt)
    epochvar=tf.Variable(0,dtype=tf.int64,trainable=False)
    best=tf.Variable(float("inf"),dtype=tf.float32,trainable=False)
    best_epoch=tf.Variable(0,dtype=tf.int64,trainable=False)
    ckpt=tf.train.Checkpoint(model=model,optimizer=opt,completed_epochs=epochvar,best_val=best,best_epoch=best_epoch)
    last_manager=tf.train.CheckpointManager(ckpt,str(output/"checkpoints/last"),max_to_keep=2)
    best_manager=tf.train.CheckpointManager(ckpt,str(output/"checkpoints/best"),max_to_keep=1)
    if a.resume and last_manager.latest_checkpoint:
        ckpt.restore(last_manager.latest_checkpoint).assert_existing_objects_matched()
        print("RESUMED",last_manager.latest_checkpoint,"epoch",int(epochvar.numpy()),flush=True)
    elif a.resume and int(epochvar.numpy()):raise RuntimeError("Missing restart checkpoint")
    manifest=dict(status="training",started=time.strftime("%Y-%m-%d %H:%M:%S"),pid=os.getpid(),
                  fold=a.fold,initialization=config["initialization"],independent_initialization=True,
                  train_val_test_disjoint=True,test_used_for_training=False,test_used_for_selection=False,
                  paper_main_table_eligible=False,model_parameters=model.count_params(),total_epochs=a.epochs,
                  configuration_sha256=sha(configpath),tensorflow=tf.__version__,precision=a.precision)
    write_json(output/"TRAINING_MANIFEST.json",manifest)
    train=Batches(split["train"],a.batch_size,a.workers,seed)
    val=Batches(split["val"],a.batch_size,a.workers,seed)
    started=time.time();epoch_times=[]
    for epoch in range(int(epochvar.numpy()),a.epochs):
        tick=time.time();sums=np.zeros(5);seen=0;overflows=0
        for step,(x,ys) in enumerate(dataset(train,epoch,True)):
            values,norm=train_step(x,ys);v=values.numpy();gn=float(norm.numpy());n=int(x.shape[0])
            overflows+=not np.isfinite(gn)
            if overflows>20:raise RuntimeError("Too many nonfinite gradients in one epoch")
            sums+=v*n;seen+=n
            if step%25==0:
                state=dict(status="training",fold=a.fold,epoch=epoch+1,total_epochs=a.epochs,
                    batch=step+1,batches=steps,train_images_seen=seen,train_images_total=len(split["train"]),
                    losses=dict(zip(("total","clf","seg","boundary","regularization"),map(float,v))),
                    gradient_norm=gn if np.isfinite(gn) else None,gradient_overflows=overflows,
                    lr=float(schedule(opt.iterations).numpy()),elapsed_epoch_seconds=time.time()-tick,
                    optimizer_iterations=int(opt.iterations.numpy()),pid=os.getpid())
                write_json(output/"progress.json",state);print(json.dumps(state),flush=True)
        assert seen==len(split["train"])
        vsums=np.zeros(5);vseen=0
        try:
            for x,ys in dataset(val,epoch,False):
                v=val_step(x,ys).numpy();n=int(x.shape[0]);vsums+=v*n;vseen+=n
        except Exception as exc:
            as_float32(model).save(str(output/"failure_model.h5"),include_optimizer=False)
            ckpt.save(str(output/"failure_checkpoint"))
            failure=dict(status="failed",phase="validation",epoch=epoch+1,error=str(exc),validation_images_completed=vseen)
            write_json(output/"FAILURE.json",failure);write_json(output/"progress.json",failure)
            manifest.update(failure);write_json(output/"TRAINING_MANIFEST.json",manifest)
            raise
        assert vseen==len(split["val"])
        means=sums/seen;vmeans=vsums/vseen
        epochvar.assign(epoch+1)
        improved=float(vmeans[0])<float(best.numpy())
        if improved:
            best.assign(vmeans[0]);best_epoch.assign(epoch+1)
            best_manager.save(checkpoint_number=epoch+1)
        last_manager.save(checkpoint_number=epoch+1)
        seconds=time.time()-tick;epoch_times.append(seconds)
        row=dict(epoch=epoch+1,seconds=seconds,train_total=float(means[0]),val_total=float(vmeans[0]),
                 train_clf=float(means[1]),train_seg=float(means[2]),train_boundary=float(means[3]),
                 val_clf=float(vmeans[1]),val_seg=float(vmeans[2]),val_boundary=float(vmeans[3]),
                 lr=float(schedule(opt.iterations).numpy()),gradient_overflows=overflows,
                 best_epoch=int(best_epoch.numpy()),best_val=float(best.numpy()),optimizer_iterations=int(opt.iterations.numpy()))
        write_json(output/"epochs"/f"epoch_{epoch+1:03d}.json",row)
        histories=[json.loads(p.read_text()) for p in sorted((output/"epochs").glob("epoch_*.json")) if int(p.stem.split("_")[1])<=epoch+1]
        with (output/"history.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=row.keys());writer.writeheader();writer.writerows(histories)
        state=dict(status="epoch_complete",fold=a.fold,completed_epochs=epoch+1,total_epochs=a.epochs,
                   remaining_seconds_estimate=float(np.median(epoch_times[-5:])*(a.epochs-epoch-1)),
                   last_checkpoint=last_manager.latest_checkpoint,best_checkpoint=best_manager.latest_checkpoint,
                   pid=os.getpid(),**row)
        write_json(output/"progress.json",state);print(json.dumps(state),flush=True)
    train.close();val.close()
    ckpt.restore(best_manager.latest_checkpoint).assert_existing_objects_matched()
    target=output/"best_model.h5";tmp=output/"best_model.tmp.h5"
    as_float32(model).save(str(tmp),include_optimizer=False);tmp.replace(target)
    manifest.update(status="training_complete",completed_epochs=a.epochs,best_epoch=int(best_epoch.numpy()),
                    best_val=float(best.numpy()),best_model=str(target),best_model_sha256=sha(target),
                    training_seconds_this_process=time.time()-started,
                    finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                    test_evaluation_pending=True)
    write_json(output/"TRAINING_MANIFEST.json",manifest)
    write_json(output/"progress.json",manifest)
    print("TRAINING COMPLETE",json.dumps(manifest),flush=True)
if __name__=="__main__":main()
