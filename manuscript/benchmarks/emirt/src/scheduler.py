"""Detached serial runner: disk evidence survives terminal disconnects.
No process outside this runner's newly created process group is ever killed.
Native fits have no resumable internal state; interrupted attempts stay explicit.
"""
from pathlib import Path
import argparse,csv,fcntl,json,os,signal,subprocess,sys,time,shutil
import psutil
import numpy as np
from src.common import atomic,sha
R=os.environ.get('DYNAMAP_RSCRIPT','Rscript')
DEADLINE=time.time()+86400*float(os.environ.get('DYNAMAP_BENCHMARK_DAYS','7'))
GIB=1024**3

def benchmark_load():
 parent=psutil.Process()
 found={}
 for child in [parent]+parent.children(recursive=True):
  try:found[child.pid]=child.memory_info().rss
  except(psutil.NoSuchProcess,psutil.AccessDenied):pass
 return sum(found.values()),list(found.items())

def verify_frozen(root):
 f=json.loads((root/'frozen.json').read_text())
 for n,h in f['files'].items():assert sha(root/n)==h,f'Source changed: {n}'

def stop_group(p):
 try:os.killpg(p.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:p.wait(timeout=10)
 except subprocess.TimeoutExpired:
  try:os.killpg(p.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  p.wait()

def measure(root,job,timeout=43200,deadline=DEADLINE):
 out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
 jp=out/'job.json';atomic(jp,job)
 env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root))
 cmd=[R,str(root/'src/emirt.R'),str(jp)] if job['model']=='EMIRT' else [sys.executable,'-m','src.neural','--job',str(jp)]
 fmt='{"wall_seconds":%e,"user_cpu_seconds":%U,"system_cpu_seconds":%S,"max_rss_kib":%M,"exit_status":%x}'
 timer=shutil.which('gtime') or shutil.which('time')
 if not timer or 'GNU' not in subprocess.run([timer,'--version'],capture_output=True,text=True).stdout:raise RuntimeError('GNU time is required (brew install gnu-time on macOS)')
 command=[timer,'-f',fmt,'-o',str(out/'gnu_time.txt'),*cmd]
 started=time.time();t0=time.monotonic();peak=0;reason=None
 atomic(out/'execution.json',dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),command=command,cpu_affinity="host_default",threads=2,host=psutil.cpu_count(),deadline=deadline))
 with (out/'run.log').open('a') as log,(out/'resources.csv').open('w',newline='') as fh:
  writer=csv.writer(fh);writer.writerow(['utc_epoch','elapsed_seconds','tree_rss_bytes','combined_benchmark_rss_bytes','host_available_bytes','load1','process_count'])
  p=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  atomic(root/'status.json',dict(status='RUNNING',job=job['id'],child_pid=p.pid,utc=time.time()))
  while p.poll() is None:
   elapsed=time.monotonic()-t0;rss=0;n=0
   try:
    parent=psutil.Process(p.pid)
    for child in [parent]+parent.children(recursive=True):
     try:rss+=child.memory_info().rss;n+=1
     except psutil.NoSuchProcess:pass
   except psutil.NoSuchProcess:pass
   combined,_=benchmark_load();peak=max(peak,rss)
   writer.writerow([time.time(),elapsed,rss,combined,psutil.virtual_memory().available,os.getloadavg()[0],n]);fh.flush()
   if time.time()>=deadline:reason='DEADLINE_CENSORED'
   elif elapsed>=timeout:reason='WALL_BUDGET_CENSORED'
   elif rss>40*GIB or combined>155*GIB:reason='MEMORY_CENSORED'
   elif shutil.disk_usage(root).free<10*GIB:reason='DISK_CENSORED'
   if reason:stop_group(p);break
   atomic(root/'status.json',dict(status='RUNNING',job=job['id'],child_pid=p.pid,elapsed_seconds=elapsed,tree_rss_gib=rss/GIB,combined_rss_gib=combined/GIB,utc=time.time()))
   time.sleep(1)
  code=p.wait()
 native={}
 if (out/'gnu_time.txt').exists():
  for line in (out/'gnu_time.txt').read_text().splitlines():
   try:native=json.loads(line)
   except ValueError:continue
 complete=code==0 and (out/'fit.json').exists()
 state=dict(status=reason or ('COMPLETE' if complete else 'FAILED'),id=job['id'],exit_code=code,wall_seconds=time.monotonic()-t0,sampled_tree_peak_rss_bytes=peak,gnu_time=native,started_epoch=started,ended_epoch=time.time())
 if complete:
  fit=json.loads((out/'fit.json').read_text());state.update(fit_sha256=sha(out/'fit.json'),validation_log_loss=fit['validation_log_loss'])
 atomic(out/'measurement.json',state);return state

def verify_completed(job,state):
 if state.get('status')!='COMPLETE':return
 if job['stage']=='selection':
  assert sha(job['out'])==state['selection_sha256'],'Selection changed'
  return
 out=Path(job['out']);assert sha(out/'fit.json')==state['fit_sha256'],'Fit changed'
 fit=json.loads((out/'fit.json').read_text())
 names={'native_fit':'native_fit.rds','predictions':'validation_predictions.csv'} if job['model']=='EMIRT' else {}
 for name,h in fit['outputs'].items():assert sha(out/names.get(name,name))==h,f'Output changed: {name}'


def select(root,job,states,jobs):
 if any(states.get(k,{}).get('status')!='COMPLETE' for k in job['dependencies']):return dict(status='DEPENDENCY_FAILED',id=job['id'])
 for dep in job['dependencies']:
  verify_completed(next(j for j in jobs if j['id']==dep),states[dep])
 scores=[]
 for cfg in job['configs']:
  deps=[j for j in jobs if j['id'] in job['dependencies'] and j['config']['id']==cfg['id']];assert len(deps)==2
  scores.append(dict(config=cfg,mean_validation_log_loss=float(np.mean([states[j['id']]['validation_log_loss'] for j in deps]))))
 best=min(scores,key=lambda z:(z['mean_validation_log_loss'],z['config']['id']))
 atomic(job['out'],dict(config=best['config'],grid=scores,test_read=False,input_fit_hashes={k:states[k]['fit_sha256'] for k in job['dependencies']}))
 return dict(status='COMPLETE',id=job['id'],selection_sha256=sha(job['out']))

def summarize(root,jobs,states):
 rows=[];targets=[]
 for j in jobs:
  if j['stage']=='selection':continue
  s=states.get(j['id'],{'status':'PENDING'})
  row={k:j.get(k) for k in ['id','country','size','model','seed','stage']};row.update(status=s['status'])
  if 'wall_seconds' in s:
   row.update(s['gnu_time']);row.update(wall_seconds=s['gnu_time'].get('wall_seconds',s['wall_seconds']),monitor_wall_seconds=s['wall_seconds'],tree_peak_rss_bytes=s['sampled_tree_peak_rss_bytes'])
  p=Path(j['out'])/'fit.json'
  if s['status']=='COMPLETE':
   f=json.loads(p.read_text());audit=f.get('data',f.get('binding',{}).get('data',{}))
   row.update({k:audit.get(k) for k in ['n_users','n_pairs','n_items','n_train','train_cells_sha256']})
   for k in ['validation_log_loss','validation_accuracy','validation_brier','parameter_count','native_converged','iterations','epochs_run','stop_reason','fit_call_seconds','training_loop_seconds']:row[k]=f.get(k)
   # Coarse EM complete-run landmarks vs neural checkpoint observations. Never
   # treat the two time conventions as an exact first-passage speedup estimate.
   obs=[]
   if j['model']=='EMIRT':obs=[(f['validation_log_loss'],s['wall_seconds'],'complete_budget_fit')]
   else:
    history=json.loads((Path(j['out'])/f"state_{f['epochs_run']:04d}.json").read_text())['history']
    overhead=max(0,s['wall_seconds']-f['seconds_this_session'])
    obs=[(h['validation_log_loss'],overhead+h['session_seconds'],'epoch_plus_nontraining_overhead') for h in history]
   for target in [.69,.67,.65,.63,.61,.59,.57,.55]:
    hits=[(t,c) for loss,t,c in obs if loss<=target]
    targets.append(dict(id=j['id'],country=j['country'],size=j['size'],model=j['model'],seed=j['seed'],stage=j['stage'],target=target,attained=bool(hits),seconds=min(hits)[0] if hits else None,time_convention=obs[0][2],status='ATTAINED' if hits else 'NOT_ATTAINED_IN_OBSERVED_RUN'))
  else:
   for target in [.69,.67,.65,.63,.61,.59,.57,.55]:
    targets.append(dict(id=j['id'],country=j['country'],size=j['size'],model=j['model'],seed=j['seed'],stage=j['stage'],target=target,attained=None,seconds=None,time_convention='unobserved',status=s['status']))
  rows.append(row)
 def table(path,rs):
  if not rs:return
  keys=list(dict.fromkeys(k for r in rs for k in r));tmp=path.with_suffix('.tmp')
  with tmp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)
  os.replace(tmp,path)
 table(root/'summary.csv',rows);table(root/'quality_targets.csv',targets)
 costs=[]
 for country in ['brazil','france','chile']:
  for model in ['EMIRT','B1','B2']:
   rr=[r for r in rows if r['country']==country and r['model']==model]
   cost=dict(country=country,model=model)
   for group,stages in [('tuning',['tuning']),('primary',['primary']),('supplementary',['landmark','strict','prior'])]:
    chosen=[r for r in rr if r['stage'] in stages]
    cost[group+'_observed_wall_seconds']=sum(r.get('wall_seconds',0) for r in chosen)
    cost[group+'_observed_cpu_seconds']=sum(r.get('user_cpu_seconds',0)+r.get('system_cpu_seconds',0) for r in chosen)
    cost[group+'_complete']=sum(r['status']=='COMPLETE' for r in chosen)
    cost[group+'_planned']=len(chosen)
   cost['workflow_observed_wall_seconds']=cost['tuning_observed_wall_seconds']+cost['primary_observed_wall_seconds']
   cost['workflow_complete']=all(r['status']=='COMPLETE' for r in rr if r['stage'] in ['tuning','primary'])
   costs.append(cost)
 table(root/'workflow_costs.csv',costs)
 contrasts=[]
 for em in rows:
  if em['model']!='EMIRT' or em['stage']!='primary' or em['status']!='COMPLETE':continue
  for dy in rows:
   if dy['stage']!='primary' or dy['status']!='COMPLETE' or dy['model']=='EMIRT':continue
   if any(dy[k]!=em[k] for k in ['country','size','seed']):continue
   assert dy['train_cells_sha256']==em['train_cells_sha256']
   gap=dy['validation_log_loss']-em['validation_log_loss']
   contrasts.append(dict(country=em['country'],size=em['size'],seed=em['seed'],dynamap=dy['model'],logloss_dynamap_minus_emirt=gap,quality_within_005=abs(gap)<=.005,wall_ratio_dynamap_over_emirt=dy['wall_seconds']/em['wall_seconds'],peak_rss_ratio_dynamap_over_emirt=dy['max_rss_kib']/em['max_rss_kib'],emirt_native_converged=em['native_converged'],dynamap_stop_reason=dy['stop_reason'],isolated_hardware=False))
 table(root/'paired_costs.csv',contrasts)
 counts={s:sum(st.get('status')==s for st in states.values()) for s in set(st.get('status') for st in states.values())}
 atomic(root/'queue_state.json',dict(utc=time.time(),n_jobs=len(jobs),counts=counts,states=states,pending=len(jobs)-len(states)))
 (root/'STATUS.md').write_text('# EM-IRT versus DYNAMAP1D\n\nStatus counts: '+json.dumps(counts)+'; pending '+str(len(jobs)-len(states))+'.\n\nSee summary.csv and quality_targets.csv. Primary, tuning, landmark, strict-tolerance and prior fits are separate. Completed is an execution status, not a convergence verdict. No test outcomes opened. Concurrent host load is recorded; no universal scalability conclusion is automatic.\n')

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();root=Path(a.root).resolve()
 if a.smoke:
  for model in ['EMIRT','B1','B2']:
   j=dict(id='smoke_'+model,model=model,seed=987,input=str(root/'data/brazil/300'),out=str(root/'tests'/model),maxit=8,tolerance=1e-6,item_variance=25,config=dict(id='smoke',optimizer='adamax',learning_rate=.01,penalty=0,batch=512,epochs=3,max_epochs=3,patience=10))
   s=measure(root,j,timeout=600,deadline=time.time()+1800);assert s['status']=='COMPLETE',s
  return
 lock=(root/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 verify_frozen(root);jobs=json.loads((root/'jobs.json').read_text());states={}
 if (root/'queue_state.json').exists():states=json.loads((root/'queue_state.json').read_text())['states']
 for j in jobs:
  if j['id'] in states:verify_completed(j,states[j['id']])
 for j in jobs:
  if j['stage']=='selection' and states.get(j['id'],{}).get('status')=='COMPLETE':
   selection=json.loads(Path(j['out']).read_text())
   for k,h in selection['input_fit_hashes'].items():assert states[k]['fit_sha256']==h
 for j in jobs:
  if j['id'] in states:continue
  verify_frozen(root)
  if time.time()>=DEADLINE:states[j['id']]=dict(status='DEADLINE_CENSORED',id=j['id']);continue
  if j['stage']=='selection':states[j['id']]=select(root,j,states,jobs);summarize(root,jobs,states);continue
  if any(states.get(k,{}).get('status')!='COMPLETE' for k in j.get('dependencies',[])):
   states[j['id']]=dict(status='DEPENDENCY_FAILED',id=j['id']);summarize(root,jobs,states);continue
  if (Path(j['out'])/'measurement.json').exists():
   states[j['id']]=json.loads((Path(j['out'])/'measurement.json').read_text());verify_completed(j,states[j['id']]);summarize(root,jobs,states);continue
  if (Path(j['out'])/'execution.json').exists():
   # Do not double-count a resumed partial fit as a fresh timing measurement.
   states[j['id']]=dict(status='INTERRUPTED_REQUIRES_REVIEW',id=j['id']);summarize(root,jobs,states);continue
  while time.time()<DEADLINE:
   rss,_=benchmark_load()
   if rss<110*GIB and shutil.disk_usage(root).free>=10*GIB:break
   atomic(root/'status.json',dict(status='WAITING_RESOURCES',job=j['id'],combined_rss_gib=rss/GIB,utc=time.time()));time.sleep(10)
  if time.time()>=DEADLINE:states[j['id']]=dict(status='DEADLINE_CENSORED',id=j['id']);continue
  print('START',j['id'],flush=True);states[j['id']]=measure(root,j);print('END',j['id'],states[j['id']]['status'],flush=True);summarize(root,jobs,states)
 summarize(root,jobs,states);atomic(root/'status.json',dict(status='QUEUE_FINISHED',utc=time.time(),counts=json.loads((root/'queue_state.json').read_text())['counts']))
if __name__=='__main__':main()
