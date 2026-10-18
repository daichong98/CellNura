"""TF2 training primitives; retain author model and repair broken composite losses."""
import sys,math
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
from common import *
sys.path.insert(0,str(SOURCE))
from models import efficent_pet_203_clf
import losses as author_losses
def configure(seed,precision):
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    gpus=tf.config.list_physical_devices("GPU")
    if not gpus:raise RuntimeError("No GPU; refusing silent CPU training")
    for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.experimental.enable_op_determinism()
    tf.keras.mixed_precision.set_global_policy(precision)
def build():
    return efficent_pet_203_clf(tf.keras.layers.Input((512,512,6),name="ip"),6,bifpn_ch=160,dropout_rate=.2,use_dropout=True)
def components(model,ys,pred):
    ys=tuple(tf.cast(v,tf.float32) for v in ys)
    pred=tuple(tf.cast(v,tf.float32) for v in pred)
    clf=author_losses.FocalLoss(ys[0],pred[0])
    seg=author_losses.FocalTverskyLoss(ys[1],pred[1])+.4*author_losses.Weighted_BCEnDice_loss(ys[1],pred[1])
    bdr=author_losses.FocalTverskyLoss(ys[2],pred[2])+.4*author_losses.Combo_loss(ys[2],pred[2])
    reg=tf.add_n([tf.cast(v,tf.float32) for v in model.losses])
    return tf.stack([clf+seg+bdr+reg,clf,seg,bdr,reg])
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,peak,total,warmup):self.peak=float(peak);self.total=int(total);self.warmup=int(warmup)
    def __call__(self,step):
        step=tf.cast(step,tf.float32)
        w=tf.cast(self.warmup,tf.float32)
        # First step strictly positive, so a successful step actually updates weights.
        ramp=self.peak*(step+1)/w
        decay=self.peak*.5*(1+tf.cos(math.pi*tf.minimum(tf.maximum((step-w)/(self.total-w),0),1)))
        return tf.where(step<w,ramp,decay)
    def get_config(self):return dict(peak=self.peak,total=self.total,warmup=self.warmup)
def optimizer(steps,epochs,precision):
    schedule=WarmupCosine(.16,steps*epochs,steps*3)
    opt=tf.keras.optimizers.SGD(learning_rate=schedule,momentum=.9,weight_decay=5e-4,jit_compile=False)
    if precision=="mixed_float16":
        opt=tf.keras.mixed_precision.LossScaleOptimizer(opt,initial_scale=1024,dynamic=True)
    return opt,schedule
def make_steps(model,opt):
    opt.build(model.trainable_variables)
    mixed=isinstance(opt,tf.keras.mixed_precision.LossScaleOptimizer)
    @tf.function(reduce_retracing=True)
    def train_step(x,ys):
        with tf.GradientTape() as tape:
            pred=model(x,training=True);values=components(model,ys,pred)
            tf.debugging.assert_all_finite(values,"Nonfinite loss")
            loss=opt.get_scaled_loss(values[0]) if mixed else values[0]
        grads=tape.gradient(loss,model.trainable_variables)
        if mixed:grads=opt.get_unscaled_gradients(grads)
        if any(g is None for g in grads):raise RuntimeError("Disconnected training branch")
        norm=tf.linalg.global_norm(grads)
        opt.apply_gradients(zip(grads,model.trainable_variables))
        return values,norm
    @tf.function(reduce_retracing=True)
    def val_step(x,ys):
        values=components(model,ys,model(x,training=False))
        tf.debugging.assert_all_finite(values,"Nonfinite validation loss")
        return values
    return train_step,val_step
def dataset(batches,epoch,training):
    sig=(tf.TensorSpec((None,512,512,6),tf.float32),
         (tf.TensorSpec((None,19),tf.float32),tf.TensorSpec((None,256,256,1),tf.float32),tf.TensorSpec((None,256,256,1),tf.float32)))
    def generate():
        # Python-side look-ahead remains asynchronous even when deterministic tf.data
        # serializes stateful from_generator operations. Sample order/seeds are unchanged.
        iterator=batches.iterate(epoch,training)
        with ThreadPoolExecutor(max_workers=1) as producer:
            pending=producer.submit(next,iterator,None)
            while True:
                batch=pending.result()
                if batch is None:break
                pending=producer.submit(next,iterator,None)
                yield batch
    ds=tf.data.Dataset.from_generator(generate,output_signature=sig)
    options=tf.data.Options();options.threading.private_threadpool_size=2
    options.experimental_deterministic=True
    return ds.with_options(options).prefetch(2)


def as_float32(model):
    """Copy learned values into a float32 inference graph; HDF5 cannot roundtrip BF16 state."""
    import numpy as np
    def clone(layer):
        cfg=layer.get_config();cfg["dtype"]="float32"
        return layer.__class__.from_config(cfg)
    converted=tf.keras.models.clone_model(model,clone_function=clone)
    values=[np.asarray(src.numpy(),dtype=dst.dtype.as_numpy_dtype) for src,dst in zip(model.weights,converted.weights)]
    assert len(model.weights)==len(converted.weights)
    converted.set_weights(values)
    for wanted,actual in zip(values,converted.weights):np.testing.assert_array_equal(wanted,actual.numpy())
    return converted
