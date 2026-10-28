"""One-dimensional DYNAMAP. Frozen architecture copied from the matched benchmark.
The embedding dimension alone changes. Political labels and test outcomes are absent.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import expit
from src.models import build_model,get_logits,get_coordinates,regularization_loss
from src.common import sha,atomic,verify_data

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
def npz_atomic(path,**payload):
 path=Path(path);tmp=path.with_suffix('.tmp.npz');np.savez(tmp,**payload);os.replace(tmp,path)
def load(input,prepared=None):
 m=verify_data(input);p=Path(input)
 with np.load(p/'train.npz',allow_pickle=False) as z:x=z['x'];y=z['y'];ui=z['ui']
 with np.load(p/'validation.npz',allow_pickle=False) as z:vx=z['x']
 val=pd.read_csv(p/'validation.csv')
 return x,y,val,vx,ui,m['n_items'],None,m['audit']
def model_sources():
 base=Path(__file__).resolve().parent
 return {str(p.relative_to(base)):sha(p) for p in sorted(base.rglob('*.py'))}
def model_for(kind,nusers,nitems,seed):return build_model(kind,nusers,nitems,1,seed)
def logits(model,kind,x):return tf.reshape(get_logits(model,kind,x),[-1])
def penalty(model,kind):return regularization_loss(model,kind)
def coords(model,kind):return get_coordinates(model,kind)
def logloss(vy,lp):
 p=np.clip(expit(lp),1e-7,1-1e-7)
 return float(np.mean(-vy*np.log(p)-(1-vy)*np.log1p(-p)))
def predict(model,kind,x):
 return np.concatenate([np.asarray(logits(model,kind,x[k:k+8192]),dtype=float) for k in range(0,len(x),8192)])
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
  lp=predict(model,kind,vx);loss=logloss(vy,lp)
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
 training_loop_seconds=time.monotonic()-started
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
 result={'status':'COMPLETE','dimension':1,'stop_reason':('validation_patience' if state['wait']>=cfg['patience'] else 'epoch_budget'),'training_loop_seconds':training_loop_seconds,'binding':binding,'kind':kind,'seed':seed,'selected_epoch':state['best_epoch'],'epochs_run':len(state['history']),'validation_log_loss':logloss(vy,lp),'validation_accuracy':float(np.mean(np.where(prob==.5,.5,(prob>.5)==vy))),'validation_brier':float(np.mean((prob-vy)**2)),'seconds_this_session':time.monotonic()-started,'parameter_count':model.count_params(),'reload_max_logit_error':error,'outputs':{name:sha(out/name) for name in ('coordinates.npz','model.weights.h5','validation_predictions.csv','best_weights.npz',f'state_{len(state["history"]):04d}.json')}}
 atomic(out/'fit.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
 job=json.loads(Path(a.job).read_text());cfg=job.get('config')
 if cfg is None:
  selection=json.loads(Path(job['selection']).read_text());assert selection['test_read'] is False
  cfg=selection['config']
 fit_one(job['model'],cfg,job['seed'],job['input'],None,job['out'])
if __name__=='__main__':main()
