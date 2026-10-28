from pathlib import Path
import hashlib,json,os

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def atomic(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def verify_data(path):
 path=Path(path);m=json.loads((path/'manifest.json').read_text())
 for name,h in m['outputs'].items():assert sha(path/name)==h,name
 assert m['test_read'] is False
 return m
