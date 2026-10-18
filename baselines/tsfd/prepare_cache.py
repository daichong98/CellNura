#!/usr/bin/env python3
"""Deterministic per-image cache; it fits no statistics and encodes no split membership."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import argparse,time
import numpy as np
from common import *
from data import preprocess,raw_arrays
_MAP={}
def init():
    import cv2;cv2.setNumThreads(1)
    for f in (1,2,3):
        _MAP[f]=tuple(np.load(CACHE/f"fold{f}_{s}.npy",mmap_mode="r+") for s in ("x","sem","bdr","tissue"))
def work(name):
    f,i=key(name);values=preprocess(name)
    for dest,val in zip(_MAP[f],values):dest[i]=val
    return name
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=16);a=ap.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True)
    signature={"split_sha256":SPLIT_SHA,"data_code_sha256":sha(HERE/"data.py"),"tissues":TISSUES,
               "deterministic_per_image_transform":True,"fitted_statistics":None}
    done=CACHE/"COMPLETE.json"
    if done.exists():
        assert json.loads(done.read_text())["signature"]==signature
        print("Cache already complete",flush=True);return
    ids=[]
    for f in (1,2,3):
        n=len(raw_arrays(f)[0])
        for suffix,shape,dtype in [("x",(n,256,256,6),"uint8"),("sem",(n,256,256),"uint8"),("bdr",(n,256,256),"uint8"),("tissue",(n,),"int16")]:
            arr=np.lib.format.open_memmap(CACHE/f"fold{f}_{suffix}.npy",mode="w+",dtype=dtype,shape=shape)
            arr.flush();del arr
        ids.extend(f"fold{f}_image_{i}" for i in range(n))
    start=time.time()
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context("spawn"),initializer=init) as pool:
        for i,name in enumerate(pool.map(work,ids,chunksize=8),1):
            if i%250==0 or i==len(ids):print(json.dumps(dict(cached=i,total=len(ids),seconds=time.time()-start)),flush=True)
    for f in (1,2,3):
        for s in ("x","sem","bdr","tissue"):
            m=np.load(CACHE/f"fold{f}_{s}.npy",mmap_mode="r+");m.flush()
    write_json(done,dict(signature=signature,images=len(ids),seconds=time.time()-start))
if __name__=="__main__":main()
