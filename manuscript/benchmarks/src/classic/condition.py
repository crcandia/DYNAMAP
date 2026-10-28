"""One native Bayesian condition. Four chains, diagnostic extensions, no test.

Children remain in this condition's process group. Original reusable chains are
read-only and copied only after their old supervisor has finished Bayesian work.
"""
import argparse,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
import psutil
from diagnostics import sha,check,diagnose
HERE=Path(__file__).resolve().parent
SEEDS=[101,201,301,401]
CHILDREN=[]

def atomic(path,obj):
 path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
 temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(temp,path)
def stop_children():
 for p in CHILDREN:
  if p.poll() is None:p.terminate()
 for p in CHILDREN:
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.kill();p.wait()
def interrupted(signum,frame):
 stop_children();raise RuntimeError('Interrupted condition, complete checkpoints retained')
def check_deadline(deadline):
 if time.time()>=deadline:raise TimeoutError('Seven-day deadline reached, complete checkpoints retained')
def run_many(commands,logs,deadline):
 procs=[];handles=[]
 try:
  for cmd,log in zip(commands,logs):
   h=Path(log).open('ab');handles.append(h);h.write(('COMMAND '+repr(cmd)+'\n').encode());h.flush()
   p=subprocess.Popen(cmd,stdout=h,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=False)
   CHILDREN.append(p);procs.append(p)
  while any(p.poll() is None for p in procs):
   check_deadline(deadline)
   if any(p.poll() not in (None,0) for p in procs):raise RuntimeError('Native child failed')
   time.sleep(2)
  if any(p.returncode for p in procs):raise RuntimeError('Native child failed')
 finally:
  if any(p.poll() is None for p in procs):stop_children()
  for h in handles:h.close()

def wait_reuse(statusfile,deadline):
 while True:
  check_deadline(deadline)
  status=json.loads(Path(statusfile).read_text())
  state=status.get('bayes_status')
  if state=='FAILED' or status.get('status')=='ERROR':raise RuntimeError('Original Bayesian pipeline failed; explicit recovery review required')
  terminal=state in ['DIAGNOSTIC_SCREEN_PASSED','NONCONVERGED_AT_ITERATION_CAP']
  active=False
  for key,job in status.get('jobs',{}).items():
   if key.startswith('bayes') and job.get('returncode') is None:
    try:
     p=psutil.Process(job['pid']);active|=p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:pass
  if terminal and not active:return status
  time.sleep(15)

def copy_reuse(source,out,prepared,runner,status):
 source=Path(source);receipt={}
 for seed in SEEDS:
  src=source/f'bayes_{seed}';dest=out/'chains'/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
  if (src/'RUNNING').exists():raise RuntimeError('Original chain lock remains; no copy/resume until reviewed')
  files=[src/'fingerprint.rds',src/'initial_state.rds']+sorted(src.glob('chunk_*.rds'))
  assert len(files)>=22
  for p in files:
   target=dest/p.name;digest=sha(p)
   if target.exists():assert sha(target)==digest
   else:
    temp=target.with_suffix(target.suffix+'.copying')
    if temp.exists():raise RuntimeError('Preserve partial legacy copy')
    # Ordinary physical copy. No hardlink can mutate the original checkpoint.
    shutil.copy2(p,temp);assert sha(temp)==digest;os.replace(temp,target)
   receipt[str(p)]=dict(sha256=digest,destination=str(target))
 atomic(out/'reuse_manifest.json',dict(original_status=status,prepared_sha256=sha(prepared),original_runner_sha256=sha(runner),copied_files=receipt))

def verify_output(folder):
 m=json.loads((folder/'manifest.json').read_text())
 check({str(folder/n):h for n,h in m['output_hashes'].items()})
 if 'checkpoint_hashes' in m:check(m['checkpoint_hashes'])
 return m

def execute(args):
 inp=Path(args.input).resolve();prepdir=Path(args.prepared).resolve();out=Path(args.output).resolve()
 prepared=prepdir/'prepared.rds';rscript=os.environ.get('DYNAMAP_RSCRIPT','Rscript')
 out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir(exist_ok=True)
 if (out/'manifest.json').exists():
  old=json.loads((out/'manifest.json').read_text());check(old['input_hashes']);check({str(out/n):h for n,h in old.get('output_hashes',{}).items()})
  if old['status'] in ['CONVERGED','NONCONVERGED_AT_ITERATION_CAP']:return old
  raise RuntimeError('Existing failed condition requires explicit review, checkpoints preserved')
 lock=out/'RUNNING'
 try:lock.mkdir()
 except FileExistsError:raise RuntimeError('Existing condition lock: inspect process before recovery')
 (lock/'pid').write_text(str(os.getpid()))
 bindings={str(p):sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ['.py','.R']}
 bindings.update({str(prepared):sha(prepared),str(prepdir/'manifest.json'):sha(prepdir/'manifest.json'),str(inp/'manifest.json'):sha(inp/'manifest.json')})
 im=json.loads((inp/'manifest.json').read_text());bindings.update({str(inp/n):h for n,h in im['outputs'].items()});check(bindings)
 record=dict(kind='bayesian',status='RUNNING',started=time.time(),input_hashes=bindings,deadline=args.deadline,test_read=False,converged=False)
 fingerprint=out/'input_fingerprint.json'
 if fingerprint.exists():assert json.loads(fingerprint.read_text())==bindings
 else:atomic(fingerprint,bindings)
 atomic(out/'status.json',record)
 runner=HERE/'run_chain.R'
 try:
  if args.reuse_chains:
   if not args.reuse_status or not args.reuse_runner:raise ValueError('All reuse arguments required')
   runner=Path(args.reuse_runner).resolve()
   if not (out/'reuse_manifest.json').exists():
    status=wait_reuse(args.reuse_status,args.deadline)
    copy_reuse(args.reuse_chains,out,prepared,runner,status)
   else:
    reuse=json.loads((out/'reuse_manifest.json').read_text())
    assert reuse['prepared_sha256']==sha(prepared) and reuse['original_runner_sha256']==sha(runner)
    for src,v in reuse['copied_files'].items():assert sha(src)==v['sha256'] and sha(v['destination'])==v['sha256']
   bindings[str(runner)]=sha(runner);bindings[str(runner.parent/'native_chunk.R')]=sha(runner.parent/'native_chunk.R')
  chains=[out/'chains'/f'seed{s}' for s in SEEDS]
  ends=[]
  for chain in chains:
   files=list(chain.glob('chunk_*.rds'));ends.append(max([int(p.stem.split('_')[1]) for p in files],default=0))
  target=max(10000,max(ends));target=((target+4999)//5000)*5000
  while target<=30000:
   check_deadline(args.deadline);check(bindings)
   commands=[[rscript,'--vanilla',str(runner),str(prepared),str(chain),str(seed),str(target)] for chain,seed in zip(chains,SEEDS)]
   run_many(commands,[out/'logs'/f'chain_{s}_{target}.log' for s in SEEDS],args.deadline)
   snaps=[out/'snapshots'/f'iter{target}'/f'seed{s}' for s in SEEDS]
   for chain,snap,seed in zip(chains,snaps,SEEDS):
    if (snap/'manifest.json').exists():verify_output(snap)
    else:run_many([[rscript,'--vanilla',str(HERE/'snapshot.R'),str(prepared),str(chain),str(target),str(snap)]],[out/'logs'/f'snapshot_{seed}_{target}.log'],args.deadline)
   diagdir=out/'diagnostics'/f'iter{target}'
   if (diagdir/'audit.json').exists():
    diag=json.loads((diagdir/'audit.json').read_text());check(diag['input_hashes']);check({str(diagdir/n):h for n,h in diag['output_hashes'].items()})
   else:diag=diagnose(snaps,diagdir)
   record.update(target=target,probability_convergence=diag['probability_convergence'],coordinate_convergence=diag['coordinate_convergence'],all_item_parameter_convergence=diag['all_item_parameter_convergence'],diagnostic_audit=str(diagdir/'audit.json'))
   atomic(out/'status.json',record)
   if diag['all_converged']:
    pred=out/'validation_integrated'
    if (pred/'manifest.json').exists():verify_output(pred)
    else:run_many([[rscript,'--vanilla',str(HERE/'posterior_predict.R'),str(prepared),str(inp),str(inp/'validation.csv'),'validation',str(diagdir/'audit.json'),str(pred)]],[out/'logs'/'validation_integrated.log'],args.deadline)
    record.update(status='CONVERGED',converged=True,probabilityfile=str(pred/'predictions.csv'),coordinatefile=str(pred/'coordinates.csv'),prediction='integrated posterior probability, all retained draws and all four chains')
    break
   if target==30000:
    record.update(status='NONCONVERGED_AT_ITERATION_CAP',converged=False,excluded_from_predictive_ranking=True);break
   target+=5000
  check(bindings)
  record.update(ended=time.time(),output_hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p not in [out/'manifest.json',out/'status.json'] and p.name!='pid' and 'logs' not in p.parts})
  atomic(out/'manifest.json',record);atomic(out/'status.json',record);return record
 except Exception as exc:
  stop_children();record.update(status='DEADLINE' if isinstance(exc,TimeoutError) else 'FAILED',error=repr(exc),ended=time.time(),converged=False,excluded_from_predictive_ranking=True)
  atomic(out/'status.json',record);raise
 finally:
  shutil.rmtree(lock)
if __name__=='__main__':
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--prepared',required=True);p.add_argument('--output',required=True);p.add_argument('--deadline',required=True,type=float)
 p.add_argument('--reuse-chains');p.add_argument('--reuse-status');p.add_argument('--reuse-runner');execute(p.parse_args())

