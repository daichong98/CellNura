"""Author preprocessing/augmentation with explicit frozen image IDs and CPU workers."""
import sys,ast,copy,zlib
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from common import *
sys.path.insert(0,str(CODE))
import tsfd_runtime as runtime
from skimage.segmentation import find_boundaries
_RAW={};_CACHED={}
def raw_arrays(f):
    if f not in _RAW:
        _RAW[f]=(np.load(DATA/f"Fold {f}/images/fold{f}/images.npy",mmap_mode="r"),
                 np.load(DATA/f"Fold {f}/masks/fold{f}/masks.npy",mmap_mode="r"),
                 np.load(DATA/f"Fold {f}/images/fold{f}/types.npy",mmap_mode="r"))
    return _RAW[f]
def preprocess(name):
    f,i=key(name);images,masks,types=raw_arrays(f)
    image=np.asarray(images[i]).astype(np.uint8)
    if runtime.UTILS is None:runtime.initialize()
    with np.errstate(divide="ignore",invalid="ignore"):
        h=runtime.UTILS.deconv_stains(image,runtime.UTILS.her_from_rgb)
    raw=masks[i]
    sem=np.argmax(raw,axis=-1).astype(np.uint8)
    sem=np.array([5,1,2,3,4,0],dtype=np.uint8)[sem]
    boundary=np.zeros((256,256),dtype=np.uint8)
    for c in range(5):
        boundary|=find_boundaries(raw[...,c],connectivity=1,mode="thick",background=0).astype(np.uint8)
    # Author's all-background special case. Do not erase valid homogeneous foreground.
    if not np.any(sem):boundary.fill(0)
    return np.concatenate((image,h),axis=-1),sem,boundary,TISSUES.index(str(types[i]))
def load_cached(name):
    f,i=key(name)
    if f not in _CACHED:
        _CACHED[f]=tuple(np.load(CACHE/f"fold{f}_{s}.npy",mmap_mode="r") for s in ("x","sem","bdr","tissue"))
    return tuple(a[i] for a in _CACHED[f])
def augment(x,sem,bdr,seed):
    # imgaug 0.4 uses np.bool internally; compatibility alias only.
    if "bool" not in np.__dict__:np.bool=np.bool_
    from imgaug import augmenters as iaa
    sometimes=lambda aug:iaa.Sometimes(.90,aug)
    seq=iaa.Sequential([iaa.SomeOf(2,[
        sometimes(iaa.Fliplr(.9)),sometimes(iaa.Flipud(.9)),
        sometimes(iaa.Affine(scale={"x":(.8,1.2),"y":(.8,1.2)},order=0,backend="cv2")),
        sometimes(iaa.Affine(translate_percent={"x":(-.2,.2),"y":(-.2,.2)},order=0,backend="cv2")),
        sometimes(iaa.Affine(rotate=(-25,25),order=0,backend="cv2")),
        sometimes(iaa.Affine(shear=(-8,8),order=0,backend="cv2")),
        sometimes(iaa.KeepSizeByResize(iaa.Crop(percent=(.05,.25),keep_size=False),interpolation="nearest"))
    ],random_order=True)],random_order=True)
    seq.seed_(int(seed));det=seq.to_deterministic()
    frame=np.ones((256,256),dtype=np.uint8);frame[2:-2,2:-2]=0
    # Same deterministic transform for each array, exactly as the author's generator.
    rgb=det.augment_image(np.array(x[...,:3]))
    her=det.augment_image(np.array(x[...,3:]))
    sem=det.augment_image(np.array(sem));bdr=det.augment_image(np.array(bdr))
    frame=det.augment_image(frame)
    bdr=((bdr>0)|((frame>0)&(sem>0))).astype(np.uint8)
    return np.concatenate([rgb,her],axis=-1),sem,bdr
def sample(args):
    name,training,seed,cached=args
    x,sem,bdr,tissue=load_cached(name) if cached else preprocess(name)
    if training:x,sem,bdr=augment(x,sem,bdr,seed)
    frame=np.ones((256,256),dtype=np.uint8);frame[2:-2,2:-2]=0
    bdr=((bdr>0)|((frame>0)&(sem>0))).astype(np.uint8)
    rgb=cv2.resize(x[...,:3],(512,512),interpolation=cv2.INTER_LINEAR)
    her=cv2.resize(x[...,3:],(512,512),interpolation=cv2.INTER_LINEAR)
    x=np.concatenate([rgb,her],axis=-1).astype(np.float32)
    tissue_onehot=np.eye(19,dtype=np.float32)[int(tissue)]
    assert sem.min()>=0 and sem.max()<=5
    return x,(tissue_onehot,np.array(sem[...,None],dtype=np.float32),bdr[...,None].astype(np.float32))
class Batches:
    def __init__(self,ids,batch,workers,seed,cached=True):
        self.ids=list(ids);self.batch=batch;self.seed=seed;self.cached=cached
        self.pool=ThreadPoolExecutor(workers);cv2.setNumThreads(1)
    def __len__(self):return (len(self.ids)+self.batch-1)//self.batch
    def iterate(self,epoch,training):
        ids=self.ids.copy()
        if training:np.random.default_rng(self.seed+epoch).shuffle(ids)
        for start in range(0,len(ids),self.batch):
            tasks=[(name,training,(zlib.crc32(name.encode())+self.seed+epoch*100003)%2147483647,self.cached) for name in ids[start:start+self.batch]]
            rows=list(self.pool.map(sample,tasks))
            yield np.stack([r[0] for r in rows]),tuple(np.stack([r[1][j] for r in rows]) for j in range(3))
    def close(self):self.pool.shutdown()
