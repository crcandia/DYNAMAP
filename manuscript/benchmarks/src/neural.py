"""Matched-support neural/spatial fits. Test is read only by explicit sealed predict.

Every model receives the exact observed cells passed to the native classics.
Six validation configurations, two tuning seeds; three new final seeds. Epoch
checkpoints include optimizer state; deterministic epoch permutations allow
continuation without changing the training order. No political labels loaded.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
from pathlib import Path
import argparse,hashlib,json,math,sys,time
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import expit
from src.models import build_model,get_logits,get_coordinates,regularization_loss

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def npz_atomic(path,**payload):
 path=Path(path);tmp=path.with_suffix('.tmp.npz');np.savez(tmp,**payload);os.replace(tmp,path)
def load(input,prepared):
 input=Path(input);prepared=Path(prepared);m=json.loads((input/'manifest.json').read_text())
 for name,digest in m['outputs'].items():assert sha(input/name)==digest,name
 people=pd.read_csv(input/'people.csv');pairs=pd.read_csv(input/'pairs.csv')
 pm=json.loads((prepared/'manifest.json').read_text());bound=pm.get('coverage_hashes')
 assert sha(prepared/'prepared.rds')==pm['prepared_sha256']
 if bound is not None:assert pm['input_manifest_sha256']==sha(input/'manifest.json')
 assert bound is not None, 'Run src/classic/prepare.R for this input first'
 for name,h in bound.items():assert sha(prepared/name)==h,name
 pc=pd.read_csv(prepared/'people_coverage.csv');jc=pd.read_csv(prepared/'pair_coverage.csv')
 for f,name,n in [(people,'person_index',m['n_people']),(pairs,'pair_index',m['n_rollcalls']),(pc,'person_index',m['n_people']),(jc,'pair_index',m['n_rollcalls'])]:
  assert np.array_equal(f[name].to_numpy(),np.arange(n)+1),name
 assert people.uuid.is_unique and people.global_user_index.is_unique
 assert np.array_equal(pc.uuid.astype(str).to_numpy(),people.uuid.astype(str).to_numpy())
 for f in (pc,jc):assert f.retained.astype(str).str.lower().isin(['true','false','1','0']).all()
 ui=pc.loc[pc.retained.astype(str).str.lower().isin(['true','1']),'person_index'].to_numpy(dtype=int)-1
 ji=jc.loc[jc.retained.astype(str).str.lower().isin(['true','1']),'pair_index'].to_numpy(dtype=int)-1
 matrix=np.memmap(input/'train_matrix.int8',dtype=np.int8,mode='r',shape=(m['n_people'],m['n_rollcalls']))
 sub=np.asarray(matrix[np.ix_(ui,ji)]);assert np.isin(sub,[-1,0,1]).all();u,j=np.nonzero(sub>=0)
 keptpairs=pairs.iloc[ji]
 x=np.column_stack([keptpairs.low_item_index.to_numpy()[j],u,keptpairs.high_item_index.to_numpy()[j]]).astype(np.int32)
 y=sub[u,j].astype(np.float32)
 nitems=int(max(pairs.low_item_index.max(),pairs.high_item_index.max())+1)
 mapping=np.full(len(people),-1,dtype=int);mapping[ui]=np.arange(len(ui));allowed=np.zeros(len(pairs),bool);allowed[ji]=True
 def query(frame):
  assert frame.source_id.notna().all() and frame.source_id.is_unique
  assert np.isfinite(frame.vote.to_numpy(float)).all() and frame.vote.isin([0,1]).all()
  pu=frame.person_index.to_numpy(int)-1;pj=frame.pair_index.to_numpy(int)-1
  assert np.all((pu>=0)&(pu<len(mapping))) and np.all((pj>=0)&(pj<len(pairs)))
  ok=(mapping[pu]>=0)&allowed[pj]
  q=frame.loc[ok].copy();p0=pairs.iloc[pj[ok]]
  xx=np.column_stack([p0.low_item_index.to_numpy(),mapping[pu[ok]],p0.high_item_index.to_numpy()]).astype(np.int32)
  return q,xx,ok
 val,vx,_=query(pd.read_csv(input/'validation.csv'))
 assert len(x)>0 and len(val)>0 and len(ui)>=4
 audit={'input_manifest':sha(input/'manifest.json'),'people_coverage':sha(prepared/'people_coverage.csv'),'pair_coverage':sha(prepared/'pair_coverage.csv'),'n_train':len(x),'n_users':len(ui),'n_items':nitems,'n_pairs':len(ji),'train_cells_sha256':hashlib.sha256(np.column_stack([ui[u]+1,ji[j]+1,y.astype(int)]).astype('<i8').tobytes()).hexdigest()}
 return x,y,val,vx,ui,nitems,query,audit

def model_sources():
 base=Path(__file__).resolve().parent
 files=[base/'neural.py',base/'models.py',base/'vendor/model.py',base/'vendor/choice_models.py']
 return {str(f.relative_to(base)):sha(f) for f in files}

def config_grid(kind):
 opts=[('adam',.003),('adam',.01)] if kind=='B3' else [('adamax',.01),('adam',.003)]
 return [{'id':f'{o}_{lr:g}_l2_{r:g}','optimizer':o,'learning_rate':lr,'penalty':r,'batch':512,'epochs':80,'max_epochs':160,'patience':10} for o,lr in opts for r in (0.,.0001,.01)]
def model_for(kind,nusers,nitems,seed):
 tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
 return build_model(kind,nusers,nitems,2,seed)
def logits(model,kind,x):
 return tf.reshape(get_logits(model,kind,x),[-1])
def penalty(model,kind):
 return regularization_loss(model,kind)
def coords(model,kind):return get_coordinates(model,kind)
def predict(model,kind,x):
 return np.concatenate([np.asarray(logits(model,kind,x[k:k+8192]),dtype=float) for k in range(0,len(x),8192)]) if len(x) else np.empty(0)

def fit_one(kind,cfg,seed,input,prepared,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 x,y,val,vx,ui,nitems,query,dataudit=load(input,prepared)
 binding={'data':dataudit,'kind':kind,'seed':seed,'configuration':cfg,'source_sha256':sha(__file__),'model_source_hashes':model_sources(),'tf_version':tf.__version__}
 if (out/'fit.json').exists():
  old=json.loads((out/'fit.json').read_text());assert old['binding']==binding
  for name,h in old['outputs'].items():assert sha(out/name)==h
  return old
 if (out/'binding.json').exists():assert json.loads((out/'binding.json').read_text())==binding
 else:atomic(out/'binding.json',binding)
 model=model_for(kind,len(ui),nitems,seed)
 opt=(tf.keras.optimizers.Adamax if cfg['optimizer']=='adamax' else tf.keras.optimizers.Adam)(learning_rate=cfg['learning_rate'])
 epoch_var=tf.Variable(0,dtype=tf.int64,trainable=False)
 checkpoint=tf.train.Checkpoint(model=model,optimizer=opt,epoch=epoch_var)
 manager=tf.train.CheckpointManager(checkpoint,str(out/'checkpoints'),max_to_keep=2)
 state={'best':None,'best_epoch':0,'wait':0,'history':[],'extended':False}
 if manager.latest_checkpoint:
  checkpoint.restore(manager.latest_checkpoint)
  state=json.loads((out/f'state_{int(epoch_var.numpy()):04d}.json').read_text())
 started=time.monotonic();first_epoch=int(epoch_var.numpy())
 @tf.function
 def train_epoch(ds):
  for bx,by in ds:
   with tf.GradientTape() as tape:
    loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=by,logits=logits(model,kind,bx)))+tf.cast(cfg['penalty'],tf.float32)*penalty(model,kind)
   grad=tape.gradient(loss,model.trainable_variables)
   opt.apply_gradients(zip(grad,model.trainable_variables))
 vy=val.vote.to_numpy(float)
 for epoch in range(first_epoch,cfg['max_epochs']):
  if state['wait']>=cfg['patience']:break
  if epoch>=cfg['epochs'] and not state['extended']:break
  order=np.random.default_rng(seed*100000+epoch).permutation(len(y))
  ds=tf.data.Dataset.from_tensor_slices((x[order],y[order])).batch(cfg['batch'])
  options=tf.data.Options();options.threading.private_threadpool_size=1;ds=ds.with_options(options)
  train_epoch(ds)
  lp=predict(model,kind,vx);loss=float(np.mean(np.logaddexp(0,lp)-vy*lp))
  if not np.isfinite(loss):raise RuntimeError('Nonfinite validation loss')
  oldbest=state['best'];improved=oldbest is None or loss<oldbest
  state['wait']=0 if oldbest is None or loss<oldbest-1e-5 else state['wait']+1
  if improved:
   state['best']=loss;state['best_epoch']=epoch+1
   npz_atomic(out/f'best_epoch_{epoch+1:04d}.npz',**{f'w{i:04d}':w for i,w in enumerate(model.get_weights())})
  state['history'].append({'epoch':epoch+1,'validation_log_loss':loss,'session_seconds':time.monotonic()-started})
  if epoch+1==cfg['epochs']:
   recent=state['history'][-11:]
   state['extended']=bool(state['best_epoch']>cfg['epochs']-10 and recent[0]['validation_log_loss']-min(r['validation_log_loss'] for r in recent[1:])>1e-4 and state['wait']<cfg['patience'])
  epoch_var.assign(epoch+1);atomic(out/f'state_{epoch+1:04d}.json',state);manager.save(checkpoint_number=epoch+1)
  atomic(out/'progress.json',{'epoch':epoch+1,'best_epoch':state['best_epoch'],'best_validation_log_loss':state['best'],'elapsed_session_seconds':time.monotonic()-started})
  print(json.dumps({'kind':kind,'seed':seed,'epoch':epoch+1,'validation':loss}),flush=True)
  if state['wait']>=cfg['patience']:break
 with np.load(out/f'best_epoch_{state["best_epoch"]:04d}.npz',allow_pickle=False) as z:
  weights=[z[k] for k in sorted(z.files)];model.set_weights(weights)
 npz_atomic(out/'best_weights.npz',**{f'w{i:04d}':w for i,w in enumerate(weights)})
 lp=predict(model,kind,vx);prob=expit(lp)
 val['probability']=prob;val['covered']=True;val.to_csv(out/'validation_predictions.csv',index=False)
 users,items=coords(model,kind)
 payload={'users':np.asarray(users),'items':np.asarray(items),'person_index':ui+1}
 if kind=='B3':payload['item_bias']=model.item_bias.numpy()
 npz_atomic(out/'coordinates.npz',**payload)
 model.save_weights(str(out/'model.weights.h5'))
 # Rebuild independently and verify saved weights produce exactly these predictions.
 reloaded=model_for(kind,len(ui),nitems,seed);reloaded.load_weights(str(out/'model.weights.h5'))
 error=float(np.max(np.abs(predict(reloaded,kind,vx)-lp)));assert error<1e-6,error
 result={'status':'COMPLETE','binding':binding,'kind':kind,'seed':seed,'selected_epoch':state['best_epoch'],'epochs_run':len(state['history']),'validation_log_loss':float(np.mean(np.logaddexp(0,lp)-vy*lp)),'seconds_this_session':time.monotonic()-started,'parameter_count':model.count_params(),'reload_max_logit_error':error,'outputs':{name:sha(out/name) for name in ('coordinates.npz','model.weights.h5','validation_predictions.csv','best_weights.npz')}}
 atomic(out/'fit.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--kind',choices=['B1','B2','B3','B4'],required=True);p.add_argument('--stage',choices=['tune','fit','predict'],required=True);p.add_argument('--selection',type=Path);p.add_argument('--seeds',default='301,401,501');p.add_argument('--fit-directory',type=Path);p.add_argument('--query',type=Path);p.add_argument('--seal',type=Path);a=p.parse_args()
 if a.stage=='tune':
  records=[]
  for cfg in config_grid(a.kind):
   fits=[fit_one(a.kind,cfg,seed,a.input,a.prepared,a.out/f'{cfg["id"]}_seed{seed}') for seed in (101,201)]
   records.append({'config':cfg,'mean_validation_log_loss':float(np.mean([f['validation_log_loss'] for f in fits]))})
  best=min(records,key=lambda r:(r['mean_validation_log_loss'],r['config']['id']))
  atomic(a.out/'selection.json',{'kind':a.kind,'selected':best,'all_configurations':records,'test_read':False,'data_manifest_sha256':sha(a.input/'manifest.json'),'tuning_manifest_path':str((a.input/'manifest.json').resolve()),'source_sha256':sha(__file__),'model_source_hashes':model_sources()})
 elif a.stage=='fit':
  selection=json.loads(a.selection.read_text());assert selection['kind']==a.kind and selection['test_read'] is False
  assert sha(selection['tuning_manifest_path'])==selection['data_manifest_sha256']
  assert json.loads(Path(selection['tuning_manifest_path']).read_text())['country']==json.loads((a.input/'manifest.json').read_text())['country']
  assert selection['source_sha256']==sha(__file__) and selection['model_source_hashes']==model_sources()
  runs=[]
  for seed in map(int,a.seeds.split(',')):
   out=a.out/f'seed{seed}';fit_one(a.kind,selection['selected']['config'],seed,a.input,a.prepared,out);runs.append(str(out.resolve()))
  atomic(a.out/'complete.json',{'kind':a.kind,'runs':runs,'selection_sha256':sha(a.selection),'test_read':False})
 else:
  assert a.seal and a.seal.exists(),'Country selection seal required before test'
  seal=json.loads(a.seal.read_text());fit=json.loads((a.fit_directory/'fit.json').read_text());key=str((a.fit_directory/'fit.json').resolve())
  assert seal['fits'][key]==sha(a.fit_directory/'fit.json'),'Fit not sealed'
  assert seal['queries'][str(a.query.resolve())]==sha(a.query),'Query not sealed'
  assert fit['kind']==a.kind and fit['binding']['model_source_hashes']==model_sources()
  for name,h in fit['outputs'].items():assert sha(a.fit_directory/name)==h
  x,y,val,vx,ui,nitems,query,audit=load(a.input,a.prepared);assert audit==fit['binding']['data']
  model=model_for(a.kind,len(ui),nitems,fit['seed']);model.load_weights(str(a.fit_directory/'model.weights.h5'))
  raw=pd.read_csv(a.query);frame,xx,ok=query(raw);raw['covered']=ok;raw['probability']=np.nan;raw.loc[ok,'probability']=expit(predict(model,a.kind,xx))
  a.out.parent.mkdir(parents=True,exist_ok=True);raw.to_csv(a.out,index=False)
  atomic(a.out.with_suffix('.manifest.json'),{'query_sha256':sha(a.query),'fit_sha256':sha(a.fit_directory/'fit.json'),'seal_sha256':sha(a.seal),'output_sha256':sha(a.out),'n_covered':int(ok.sum())})
if __name__=='__main__':main()

