"""Materialize identical effective training cells, without reading test or labels.
The old preparations are read only. New inputs include matrix and triplet views of
exactly the same observations; source and output hashes bind every measurement.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from src.common import sha,atomic,verify_data

def prepare(old,out,country,size):
 old=Path(old);out=Path(out)/country/size;i=old/'nativeinputs'/country/size
 p=old/'prepared'/country/size/'base'
 m=json.loads((i/'manifest.json').read_text());pm=json.loads((p/'manifest.json').read_text())
 for n,h in m['outputs'].items():assert sha(i/n)==h
 assert sha(p/'prepared.rds')==pm['prepared_sha256']
 bound=pm.get('coverage_hashes')
 assert bound is not None
 assert pm['input_manifest_sha256']==sha(i/'manifest.json')
 for n,h in bound.items():assert sha(p/n)==h
 pc=pd.read_csv(p/'people_coverage.csv');jc=pd.read_csv(p/'pair_coverage.csv')
 people=pd.read_csv(i/'people.csv');pairs=pd.read_csv(i/'pairs.csv')
 assert np.array_equal(pc.uuid.astype(str),people.uuid.astype(str))
 for frame,name,n in [(pc,'person_index',m['n_people']),(jc,'pair_index',m['n_rollcalls']),(people,'person_index',m['n_people']),(pairs,'pair_index',m['n_rollcalls'])]:assert np.array_equal(frame[name],np.arange(n)+1)
 keep=lambda f:f.retained.astype(str).str.lower().isin(['true','1']).to_numpy()
 for f in [pc,jc]:assert f.retained.astype(str).str.lower().isin(['true','false','1','0']).all()
 ui=np.flatnonzero(keep(pc));ji=np.flatnonzero(keep(jc))
 raw=np.memmap(i/'train_matrix.int8',dtype=np.int8,mode='r',shape=(m['n_people'],m['n_rollcalls']))
 v=np.asarray(raw[np.ix_(ui,ji)]);assert np.isin(v,[-1,0,1]).all()
 u,j=np.nonzero(v>=0);y=v[u,j].astype(np.float32)
 kp=pairs.iloc[ji];x=np.column_stack([kp.low_item_index.to_numpy()[j],u,kp.high_item_index.to_numpy()[j]]).astype(np.int32)
 ph=np.full(m['n_people'],-1,int);ph[ui]=np.arange(len(ui));jh=np.full(m['n_rollcalls'],-1,int);jh[ji]=np.arange(len(ji))
 val=pd.read_csv(i/'validation.csv');assert val.source_id.is_unique and val.vote.isin([0,1]).all()
 vu=val.person_index.to_numpy(int)-1;vj=val.pair_index.to_numpy(int)-1
 ok=(ph[vu]>=0)&(jh[vj]>=0);val=val.loc[ok].copy();vu=vu[ok];vj=vj[ok]
 val['local_user']=ph[vu];val['local_pair']=jh[vj]
 assert np.all(v[ph[vu],jh[vj]]==-1),'Validation overlaps training'
 pp=pairs.iloc[vj];vx=np.column_stack([pp.low_item_index.to_numpy(),ph[vu],pp.high_item_index.to_numpy()]).astype(np.int32)
 audit={'input_manifest':sha(i/'manifest.json'),'people_coverage':sha(p/'people_coverage.csv'),'pair_coverage':sha(p/'pair_coverage.csv'),'n_train':len(y),'n_users':len(ui),'n_items':int(max(pairs.low_item_index.max(),pairs.high_item_index.max())+1),'n_pairs':len(ji),'train_cells_sha256':hashlib.sha256(np.column_stack([ui[u]+1,ji[j]+1,y.astype(int)]).astype('<i8').tobytes()).hexdigest()}
 if (out/'manifest.json').exists():assert verify_data(out)['audit']==audit;return
 out.mkdir(parents=True,exist_ok=True);v.tofile(out/'train_matrix.int8')
 np.savez(out/'train.npz',x=x,y=y,ui=ui,ji=ji);np.savez(out/'validation.npz',x=vx)
 val.to_csv(out/'validation.csv',index=False);people.iloc[ui].to_csv(out/'people.csv',index=False);kp.to_csv(out/'pairs.csv',index=False)
 files=['train_matrix.int8','train.npz','validation.npz','validation.csv','people.csv','pairs.csv']
 atomic(out/'manifest.json',dict(country=country,size=size,dimension=1,n_people=len(ui),n_pairs=len(ji),n_items=audit['n_items'],n_train=len(y),n_validation=len(val),test_read=False,audit=audit,source_prepared_sha256=pm['prepared_sha256'],outputs={n:sha(out/n) for n in files}))
 print(country,size,len(ui),len(ji),len(y),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--country',choices=['chile','france','brazil']);a=p.parse_args()
 for c,sizes in [('chile',['300','1000','5000','15000','full']),('france',['300','1000','full']),('brazil',['300','full'])]:
  if a.country and c!=a.country:continue
  for s in sizes:prepare(a.source,a.out,c,s)
