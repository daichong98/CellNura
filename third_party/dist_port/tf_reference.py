"""CPU TF2 primitive reference for the port; not execution of full old TF1 project."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import numpy as np
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
data = np.load(sys.argv[1])
x = tf.constant(data["input"].transpose(0,2,3,1))
def array(key): return tf.constant(data[key])
def block(x, prefix, training):
    for i in range(2):
        p = prefix+"."+str(i)+"."
        w = tf.transpose(array(p+"conv.weight"), [2,3,1,0])
        x = tf.nn.conv2d(x, w, strides=1, padding="VALID")
        if training:
            mean,var = tf.nn.moments(x, [0,1,2])
        else:
            mean,var = array(p+"bn.moving_mean"),array(p+"bn.moving_var")
        x = tf.nn.batch_normalization(x, mean, var, array(p+"bn.beta"),array(p+"bn.gamma"),1e-5)
        x = tf.nn.relu(tf.nn.bias_add(x,array(p+"bias")))
    return x
def forward(training):
    y=x
    skips=[]
    for i in range(5):
        if i: y=tf.nn.max_pool2d(y,2,2,"VALID")
        y=block(y,"enc."+str(i),training)
        skips.append(y)
    for i, skip in enumerate(reversed(skips[:-1])):
        p="up."+str(i)+"."
        w=tf.transpose(array(p+"weight"),[2,3,1,0])
        shape=y.shape
        y=tf.nn.conv2d_transpose(y,w,[shape[0],shape[1]*2,shape[2]*2,shape[3]//2],[1,2,2,1],"VALID")
        y=tf.nn.relu(tf.nn.bias_add(y,array(p+"bias")))
        dy,dx=(skip.shape[1]-y.shape[1])//2,(skip.shape[2]-y.shape[2])//2
        y=tf.concat([skip[:,dy:dy+y.shape[1],dx:dx+y.shape[2],:],y],axis=3)
        y=block(y,"dec."+str(i),training)
    y=tf.nn.conv2d(y,tf.transpose(array("head.weight"),[2,3,1,0]),1,"VALID")
    return tf.nn.bias_add(y,array("head.bias")).numpy().transpose(0,3,1,2)
# Separately verify the custom BN's train/eval state with TF EMA operations.
v=tf.constant(data["bn_input"].transpose(0,2,3,1))
m,var=tf.nn.moments(v,[0,1,2])
ema=tf.train.ExponentialMovingAverage(decay=.9)
# In eager mode EMA initializes directly from tensor. Graph mode original shadows
# start at zero for moment tensors; explicitly reproduce the two update equations.
mm=.1*m
vv=.1*var
mm=.9*mm+.1*m
vv=.9*vv+.1*var
bn_train=tf.nn.batch_normalization(v,m,var,array("bn_beta"),array("bn_gamma"),1e-5)
bn_eval=tf.nn.batch_normalization(v,mm,vv,array("bn_beta"),array("bn_gamma"),1e-5)
np.savez(sys.argv[2], train=forward(True), eval=forward(False),
         bn_train=bn_train.numpy().transpose(0,3,1,2),bn_eval=bn_eval.numpy().transpose(0,3,1,2),
         moving_mean=mm.numpy(),moving_var=vv.numpy())
