"""Freeze a transparent queue. No outcome-based additions or test reads."""
from pathlib import Path
import argparse,itertools
from src.common import atomic,sha

def build(root,country_filter=None):
 root=Path(root).resolve();jobs=[]
 def add(country,size,model,seed,stage='primary',**kw):
  tag=kw.pop('tag',stage);name=f'{country}_{size}_{model}_{seed}_{tag}'
  j=dict(id=name,country=country,size=size,model=model,seed=seed,stage=stage,input=str(root/'data'/country/size),out=str(root/'results'/name),**kw)
  if model=='EMIRT':j=dict(maxit=5000,tolerance=1e-6,item_variance=25,**j)
  jobs.append(j);return name
 countries=[('brazil',['300','full'],'full'),('france',['300','1000','full'],'1000'),('chile',['300','1000','5000','15000','full'],'5000')]
 for country,sizes,tunesize in countries:
  if country_filter and country!=country_filter:continue
  for model in ['B2','B1']:
   cfgs=[];dependencies=[]
   for (opt,lr),pen in itertools.product([('adamax',.01),('adam',.003)],[0.,.0001,.01]):
    cfg=dict(id=f'{opt}_{lr:g}_l2_{pen:g}',optimizer=opt,learning_rate=lr,penalty=pen,batch=512,epochs=80,max_epochs=160,patience=10)
    cfgs.append(cfg)
    for seed in [101,201]:dependencies.append(add(country,tunesize,model,seed,'tuning',tag=cfg['id'],config=cfg))
   jobs.append(dict(id=f'select_{country}_{model}',stage='selection',country=country,model=model,dependencies=dependencies,configs=cfgs,out=str(root/'selections'/f'{country}_{model}.json')))
  for si,size in enumerate(sizes):
   for ri,seed in enumerate([301,401,501]):
    models=['EMIRT','B1','B2'];offset=(si+ri)%3;models=models[offset:]+models[:offset]
    for model in models:
     kw={} if model=='EMIRT' else dict(selection=str(root/'selections'/f'{country}_{model}.json'),dependencies=[f'select_{country}_{model}'])
     add(country,size,model,seed,**kw)
 # Supplementary work follows all primary conditions, so it cannot delay them.
 for country,sizes,tunesize in countries:
  if country_filter and country!=country_filter:continue
  for size in sizes:
   for budget in [50,200]:
    add(country,size,'EMIRT',301,'landmark',tag=f'budget{budget}');jobs[-1]['maxit']=budget
   add(country,size,'EMIRT',301,'strict',tag='strict1e8');jobs[-1]['tolerance']=1e-8
  add(country,tunesize,'EMIRT',301,'prior',tag='prior6p25');jobs[-1]['item_variance']=6.25
 atomic(root/'jobs.json',jobs)
 files=list((root/'src').rglob('*.py'))+list((root/'src').rglob('*.R'))+[root/'PROTOCOL.md',root/'jobs.json']
 atomic(root/'frozen.json',{'files':{str(p.relative_to(root)):sha(p) for p in sorted(files)},'n_jobs':len(jobs)})
 print(len(jobs),'jobs',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--country',choices=['chile','france','brazil']);a=p.parse_args();build(a.root,a.country)
