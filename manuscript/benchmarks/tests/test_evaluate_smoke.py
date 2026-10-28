import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
sys.dont_write_bytecode=True
import importlib.util,tempfile,json
from pathlib import Path
import numpy as np,pandas as pd
s=importlib.util.spec_from_file_location('ev',str(Path(__file__).resolve().parents[1]/'src/evaluate.py')); e=importlib.util.module_from_spec(s); s.loader.exec_module(e)
with tempfile.TemporaryDirectory() as td:
 r=Path(td); prep=r/'nativeinputs/x/10'; prep.mkdir(parents=True); (r/'evaluation/x/10').mkdir(parents=True); (r/'data/x').mkdir(parents=True)
 pd.DataFrame({'person_index':range(1,21),'global_user_index':range(20)}).to_csv(prep/'people.csv',index=False)
 test=pd.DataFrame({'source_id':range(40),'person_index':np.repeat(range(1,21),2),'vote':np.tile([0,1],20)})
 test.to_csv(r/'evaluation/x/10/test.csv',index=False)
 pd.DataFrame({'user_index':range(20),'label':np.arange(20)%2,'fold':np.arange(20)//2}).to_csv(r/'data/x/external_cohort.csv',index=False)
 records=[]
 for model in ['B1','B2']:
  for seed in [1,2]:
   f=test.copy(); f['probability']=np.where(f.vote,.8 if model=='B1' else .6,.2 if model=='B1' else .4); pf=r/f'{model}{seed}.csv'; f.to_csv(pf,index=False)
   cf=r/f'{model}{seed}.npz'; np.savez(cf,users=np.c_[np.arange(20)%2,np.arange(20)],user_indices=np.arange(20))
   records.append(dict(model=model,seed=seed,status='complete',probabilityfile=str(pf),coordinatefile=str(cf)))
 a=e.evaluate_condition(r,'x',10,prep,records,r/'result')
 assert a['common_rows']==40
 c=pd.read_csv(r/'result/paired_contrasts.csv'); assert c.loc[c.metric=='log_loss','a_minus_b'].iloc[0]<0
 assert all(e.metrics([0,1],[.5,.5])['accuracy']==.5)
 assert e.paired_interval([0]*20)==(0.,0.,0.)
 try:e.evaluate_condition(r,'x',10,prep,records,r/'result')
 except ValueError:pass
 else:raise AssertionError('overwrite permitted')
 e.collect_conditions(r,r/'collected')
 f=pd.read_csv(r/'B11.csv'); f.loc[0,'vote']=1; f.to_csv(r/'B11.csv',index=False)
 try:e.evaluate_condition(r,'x',10,prep,records,r/'bad')
 except ValueError as ex: assert 'Target/ID' in str(ex)
 else:raise AssertionError('mismatched truth allowed')
 print('PASS: matching, 4 fits, 10 external folds, seed metrics, bootstrap sign, ties, output guard, collector, target guard')

