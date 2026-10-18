#!/usr/bin/env python3
"""Explicitly fetch a pinned upstream source. Does not fetch data, weights or environments."""
import argparse,hashlib,json,os,shutil,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('name',choices=['tsfd'])
    ap.add_argument('--destination',type=Path,default=ROOT/'third_party/tsfd')
    a=ap.parse_args(argv)
    meta=next(x for x in json.loads((ROOT/'third_party/DEPENDENCIES.json').read_text()) if x['name']==a.name)
    dest=a.destination.expanduser().resolve()
    if dest.exists():raise FileExistsError(f'Destination exists; refusing to overwrite source: {dest}')
    with tempfile.TemporaryDirectory(prefix='cellnura-source-') as tmp:
        repository=Path(tmp)/'repository'
        subprocess.run(['git','clone','--no-checkout',meta['upstream_url'],str(repository)],check=True)
        subprocess.run(['git','-C',str(repository),'checkout','--detach',meta['commit']],check=True)
        commit=subprocess.check_output(['git','-C',str(repository),'rev-parse','HEAD'],text=True).strip()
        if commit!=meta['commit']:raise ValueError('Fetched commit does not match pinned source')
        dest.mkdir(parents=True,exist_ok=False);files={}
        for directory,dirs,names in os.walk(repository):
            dirs[:]=[d for d in dirs if d not in {'.git','__pycache__','weights','data','datasets','results','logs','wandb'}]
            for name in names:
                path=Path(directory)/name
                if path.suffix.lower() not in {'.py','.txt','.md','.json','.yaml','.yml','.toml','.cfg'} and name not in {'LICENSE','LICENSE.txt','COPYING','NOTICE'}:continue
                target=dest/path.relative_to(repository);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
                files[str(target.relative_to(dest))]=hashlib.sha256(target.read_bytes()).hexdigest()
        missing=[f for f in ['scripts/models.py','scripts/bifpn.py','scripts/losses.py','slide_inference/utils.py'] if not(dest/f).is_file()]
        if missing:raise ValueError('Pinned repository has missing required source: '+str(missing))
        licenses=[f for f in files if Path(f).name in {'LICENSE','LICENSE.txt','COPYING'}]
        (dest/'FETCH_MANIFEST.json').write_text(json.dumps({'upstream_url':meta['upstream_url'],'commit':commit,'files':files,'license_files':licenses,'license_note':'No license found at collected commit; consult authors for redistribution permissions' if not licenses else 'Original upstream license files preserved'},indent=2))
    print(json.dumps({'fetched':True,'commit':commit,'files':len(files),'destination':str(dest),'weights_downloaded':False}))
    return 0
if __name__=='__main__':raise SystemExit(main())
