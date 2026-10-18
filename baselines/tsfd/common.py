"""Frozen split and source provenance; this module never imports TensorFlow."""
from pathlib import Path
import csv,hashlib,json,re,os
from cellnura_release.baseline_runtime import resource
HERE=Path(__file__).resolve().parent
METHOD=HERE
REV=Path(resource('project_root'))
ROOT=REV
CODE=HERE
SOURCE=Path(resource('tsfd_source'))/'scripts'
DATA=Path(resource('pannuke'))
SPLIT=Path(resource('split_manifest'))
SPLIT_SHA="38b39de78b4facdd2b6357a473017c817addd15714cddd7d38e5d2a0c166ce65"
CACHE=Path(resource("tsfd_cache"))
TISSUES=['Adrenal_gland','Bile-duct','Bladder','Breast','Cervix','Colon','Esophagus','HeadNeck','Kidney','Liver','Lung','Ovarian','Pancreatic','Prostate','Skin','Stomach','Testis','Thyroid','Uterus']
EXPECTED={1:(4458,787,2656),2:(4571,807,2523),3:(4402,777,2722)}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
    tmp.replace(path)
def key(name):
    m=re.fullmatch(r"fold([123])_image_(\d+)",name)
    if m is None:raise ValueError(name)
    return int(m[1]),int(m[2])
def splits(fold):
    assert sha(SPLIT)==SPLIT_SHA,"The frozen split changed"
    rows=[r for r in csv.DictReader(SPLIT.open(encoding="utf-8-sig")) if r["dataset"]=="PanNuke" and r["outer_fold"]==f"fold{fold}"]
    result={s:[r["image_id"] for r in rows if r["split"]==s] for s in ("train","val","test")}
    assert tuple(map(len,result.values()))==EXPECTED[fold]
    for s,ids in result.items():assert len(ids)==len(set(ids))
    assert not(set(result["train"])&set(result["val"]))
    assert not(set(result["train"])&set(result["test"]))
    assert not(set(result["val"])&set(result["test"]))
    assert all(key(x)[0]!=fold for x in result["train"]+result["val"])
    assert set(result["test"])=={f"fold{fold}_image_{i}" for i in range(EXPECTED[fold][2])}
    return result
def sources():
    paths=[SOURCE/x for x in ("models.py","bifpn.py","losses.py","data_gen.py","main_tsfd.py")]
    paths+=sorted(HERE.glob("*.py"))
    paths+=list((HERE/"references").glob("*.py"))
    return {str(p.name):sha(p) for p in paths}
