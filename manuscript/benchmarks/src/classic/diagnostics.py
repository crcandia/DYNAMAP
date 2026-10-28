"""Four-chain rank Rhat, bulk/tail ESS. Separate geometry and probability gates."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import arviz as az
import xarray as xr

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def check(bound):
 for p,h in bound.items():
  if sha(p)!=h:raise ValueError('Changed input '+p)
def diagnose(snapshots,out):
 out=Path(out);snapshots=[Path(p).resolve() for p in snapshots]
 if out.exists():raise RuntimeError('Preserve existing diagnostics')
 ms=[json.loads((p/'manifest.json').read_text()) for p in snapshots]
 assert len(ms)==4 and {m['seed'] for m in ms}=={101,201,301,401}
 for key in ['target','draws','burnin','thin','prepared_sha256','input_manifest_sha256','prior_variant']:
  assert len({m[key] for m in ms})==1,('Incompatible chains',key)
 assert all(m['status']=='COMPLETE' and m['burnin']==5000 and m['thin']==10 and not m['test_read'] for m in ms)
 assert ms[0]['draws']==(ms[0]['target']-5000)//10
 bound={str(Path(__file__).resolve()):sha(__file__)};ref=None
 for p,m in zip(snapshots,ms):
  bound[str(p/'manifest.json')]=sha(p/'manifest.json')
  bound.update({str(p/n):h for n,h in m['output_hashes'].items()});bound.update(m['checkpoint_hashes'])
  bound[m['prepared_path']]=m['prepared_sha256'];bound[str(Path(m['chain_path'])/'fingerprint.rds')]=m['fingerprint_sha256']
  probes=json.loads((p/'probes.json').read_text())
  if ref is None:ref=probes
  else:assert ref==probes
 check(bound);out.mkdir(parents=True)
 report=dict(snapshot_paths=[str(p) for p in snapshots],snapshot_manifest_hashes=[sha(p/'manifest.json') for p in snapshots],target=ms[0]['target'],draws_per_chain=ms[0]['draws'],n_chains=4,prepared_sha256=ms[0]['prepared_sha256'],prior_variant=ms[0]['prior_variant'],method='ArviZ rank-normalized split/folded Rhat, bulk and tail ESS',arviz_version=az.__version__,input_hashes=bound,test_read=False)
 for family in ['probabilities','coordinates','item_parameters']:
  names=ref[family];n=len(names);draws=ms[0]['draws'];mm=[]
  for p in snapshots:
   file=p/(family+'.f64');assert file.stat().st_size==draws*n*8
   mm.append(np.memmap(file,dtype='<f8',mode='r',shape=(draws,n)))
  rows=[]
  for start in range(0,n,128):
   z=np.stack([np.asarray(a[:,start:start+128]) for a in mm]);assert np.isfinite(z).all()
   constant=np.ptp(z.reshape(-1,z.shape[2]),axis=0)==0;varying=~constant
   rh=np.full(z.shape[2],np.nan);bulk=rh.copy();tail=rh.copy()
   if varying.any():
    ds=xr.Dataset({'probe':(('chain','draw','probe_id'),z[:,:,varying])})
    rh[varying]=np.asarray(az.rhat(ds,method='rank')['probe']);bulk[varying]=np.asarray(az.ess(ds,method='bulk')['probe']);tail[varying]=np.asarray(az.ess(ds,method='tail')['probe'])
   for j in range(z.shape[2]):rows.append(dict(probe=names[start+j],globally_constant=bool(constant[j]),constant_value=float(z[0,0,j]) if constant[j] else None,rank_split_rhat=rh[j],bulk_ess=bulk[j],tail_ess=tail[j]))
  f=pd.DataFrame(rows);f.to_csv(out/(family+'_diagnostics.csv'),index=False);v=f[~f.globally_constant]
  passed=bool(len(v) and np.isfinite(v[['rank_split_rhat','bulk_ess','tail_ess']]).all().all() and (v.rank_split_rhat<=1.01).all() and (v.bulk_ess>=400).all() and (v.tail_ess>=400).all())
  constants=int(f.globally_constant.sum())
  if family!='probabilities' and constants:passed=False
  # Constant predicted probabilities contribute no mixing evidence. Explicitly
  # report their exclusion, never claim those probes themselves passed.
  extrema=lambda k,how:float(getattr(v[k],how)()) if len(v) and np.isfinite(v[k]).all() else None
  report[family]=dict(passed=passed,status=('PASS_VARIABLE_PROBES_WITH_CONSTANTS' if constants else 'PASS') if passed else 'NOT_ESTABLISHED',total_probes=n,constant_probes=constants,variable_probes=len(v),max_rhat=extrema('rank_split_rhat','max'),min_bulk_ess=extrema('bulk_ess','min'),min_tail_ess=extrema('tail_ess','min'))
 report['probability_convergence']=report['probabilities']['passed'];report['coordinate_convergence']=report['coordinates']['passed'];report['all_item_parameter_convergence']=report['item_parameters']['passed']
 report['all_converged']=all(report[k]['passed'] for k in ['probabilities','coordinates','item_parameters'])
 report['next_target']=None if report['all_converged'] or report['target']>=30000 else report['target']+5000
 report['cap_reached_without_joint_convergence']=not report['all_converged'] and report['target']==30000
 report['scope']='fixed training probability and nonanchor person-coordinate probes plus ALL pair item parameters; not proof of every posterior functional'
 check(bound);report['output_hashes']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
 temp=out/'audit.json.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.rename(out/'audit.json');return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('snapshots',nargs=4,type=Path);a=p.parse_args();diagnose(a.snapshots,a.out)

