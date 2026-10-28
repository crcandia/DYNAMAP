"""Post-selection test handoff, paired evaluation and honest partial-run reports."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluate import evaluate_condition,retained_model_rows
SIZES={'chile':['300','1000','5000','15000','full'],'france':['300','1000','full'],'brazil':['300','full']}
MID={'chile':'5000','france':'1000','brazil':'full'}
KINDS=['B1','B2','B3','B4']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
 t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def checked_manifest(path,key='outputs'):
 path=Path(path);m=json.loads(path.read_text())
 for n,h in m.get(key,{}).items():assert sha(path.parent/n)==h,(path,n)
 return m

def prep(root,c,s):return root/'prepared'/c/s/'base'
def nativeinput(root,c,s):return root/'nativeinputs'/c/s

def seal(root,c):
 out=root/'results/seals'/f'{c}.json'
 fits={};selections={};inventories={};unavailable=[]
 for kind in KINDS:
  sel=root/'results/tuning'/c/kind/'selection.json'
  if sel.exists():selections[str(sel.resolve())]=sha(sel)
  else:unavailable.append({'kind':kind,'reason':'tuning did not complete'})
  for size in SIZES[c]:
   complete=root/'results/neural'/c/size/kind/'complete.json'
   if not complete.exists():
    unavailable.append({'kind':kind,'size':size,'reason':'final three-seed fit did not complete'});continue
   m=json.loads(complete.read_text());inventories[str(complete.resolve())]=sha(complete);assert m['selection_sha256']==sha(sel)
   assert len(m['runs'])==3
   seeds=[]
   for folder in m['runs']:
    p=Path(folder)/'fit.json';f=checked_manifest(p);seeds.append(f['seed']);fits[str(p.resolve())]=sha(p)
   assert sorted(seeds)==[301,401,501]
 payload={'country':c,'test_authorized':True,'created':time.time(),'fits':fits,'inventories':inventories,'selections':selections,'queries':{str((root/'evaluation'/c/s/'test.csv').resolve()):sha(root/'evaluation'/c/s/'test.csv') for s in SIZES[c]},'sources':{str(p.resolve()):sha(p) for p in (root/'src').rglob('*') if p.is_file() and p.suffix in ['.py','.R']},'protocol_sha256':sha(root/'PROTOCOL.md'),'unavailable':unavailable,'scope':'all neural selections frozen before test; native choices fixed a priori; no test-dependent extensions or model selection'}
 if out.exists():
  old=json.loads(out.read_text());assert all(old[k]==payload[k] for k in ['fits','inventories','selections','queries','sources','protocol_sha256']);return
 atomic(out,payload)

def run(root,cmd,log):
 log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('ab') as f:subprocess.run([str(x) for x in cmd],cwd=root,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)

def evaluate(root,c,s):
 out=root/'results/evaluation'/c/s
 sealfile=root/'results/seals'/f'{c}.json';sealed=json.loads(sealfile.read_text())
 for path,h in {**sealed['fits'],**sealed['inventories'],**sealed['selections'],**sealed['queries'],**sealed['sources']}.items():assert sha(path)==h,path
 assert sha(root/'PROTOCOL.md')==sealed['protocol_sha256']
 if (out/'audit.json').exists():
  cached=checked_manifest(out/'audit.json')
  for path,h in cached.get('inputs',{}).items():assert sha(path)==h,path
  assert cached['country_seal_sha256']==sha(sealfile),'Cached evaluation country seal changed'
  for path,h in cached.get('test_seal_hashes',{}).items():assert sha(path)==h,path
  return cached
 input=root/'nativeinputs'/c/s;prepared=prep(root,c,s);query=root/'evaluation'/c/s/'test.csv';post=root/'results/test_predictions'/c/s;post.mkdir(parents=True,exist_ok=True)
 records=[];people=pd.read_csv(input/'people.csv');rscript=os.environ['DYNAMAP_RSCRIPT']
 for kind in KINDS:
  completed=root/'results/neural'/c/s/kind/'complete.json'
  if not completed.exists():records.append({'model':kind,'seed':'all','status':'FAILED_OR_INCOMPLETE'});continue
  cm=json.loads(completed.read_text())
  for folder in map(Path,cm['runs']):
   fm=checked_manifest(folder/'fit.json');seed=fm['seed'];prediction=post/f'{kind}_{seed}.csv'
   if not prediction.exists():run(root,[sys.executable,'-m','src.neural','--input',input,'--prepared',prepared,'--out',prediction,'--kind',kind,'--stage','predict','--fit-directory',folder,'--query',query,'--seal',sealfile],post/f'{kind}_{seed}.log')
   pm=json.loads(prediction.with_suffix('.manifest.json').read_text());assert pm['output_sha256']==sha(prediction) and pm['query_sha256']==sha(query) and pm['fit_sha256']==sha(folder/'fit.json')
   with np.load(folder/'coordinates.npz',allow_pickle=False) as z:ids=people.iloc[z['person_index'].astype(int)-1].global_user_index.to_numpy(int).tolist()
   records.append({'model':kind,'seed':seed,'status':'COMPLETE','probabilityfile':str(prediction),'coordinatefile':str(folder/'coordinates.npz'),'user_indices':ids,'artifact_hashes':{str(folder/'fit.json'):sha(folder/'fit.json'),str(folder/'coordinates.npz'):fm['outputs']['coordinates.npz'],str(prediction):sha(prediction)}})
 wn=root/'results/wnominate'/c/s
 if (wn/'manifest.json').exists():
  wm=checked_manifest(wn/'manifest.json','output_hashes');prediction=post/'wnominate.csv';ws=post/'wnominate_test_seal.json'
  atomic(ws,{'test_authorized':True,'query_sha256':sha(query),'fit_manifest_sha256':sha(wn/'manifest.json'),'country_seal_sha256':sha(sealfile)})
  if not prediction.exists():run(root,[rscript,'--vanilla',root/'src/classic/wnominate_predict.R',wn,query,prediction,ws],post/'wnominate.log')
  pm=json.loads(Path(str(prediction)+'.manifest.json').read_text());assert pm['output_sha256']==sha(prediction) and pm['query_sha256']==sha(query) and pm['fit_manifest_sha256']==sha(wn/'manifest.json')
  f=pd.read_csv(wn/'coordinates.csv');cols=['x1','x2'] if 'x1' in f else ['coordinate1','coordinate2']
  coord=post/'wnominate_coordinates.csv';f=f.loc[np.isfinite(f[cols].to_numpy(float)).all(axis=1)].copy();f.rename(columns=dict(zip(cols,['x1','x2']))).to_csv(coord,index=False)
  records.append({'model':'W-NOMINATE','seed':'native_point','status':'COMPLETE','probabilityfile':str(prediction),'coordinatefile':str(coord),'artifact_hashes':{str(wn/'manifest.json'):sha(wn/'manifest.json'),str(prediction):sha(prediction),str(coord):sha(coord)}})
 else:records.append({'model':'W-NOMINATE','seed':'native_point','status':'FAILED_OR_INCOMPLETE'})
 bayes=root/'results/classic'/c/s/'base';bm=json.loads((bayes/'manifest.json').read_text()) if (bayes/'manifest.json').exists() else json.loads((bayes/'status.json').read_text()) if (bayes/'status.json').exists() else {'status':'NOT_COMPLETED'}
 if bm.get('converged') is True:
  diag=Path(bm['diagnostic_audit']);bs=post/'bayesian_test_seal.json';atomic(bs,{'test_authorized':True,'query_sha256':sha(query),'diagnostic_audit_sha256':sha(diag),'prepared_sha256':sha(prepared/'prepared.rds'),'country_seal_sha256':sha(sealfile)})
  dest=post/'bayesian_integrated'
  if not (dest/'manifest.json').exists():run(root,[rscript,'--vanilla',root/'src/classic/posterior_predict.R',prepared/'prepared.rds',nativeinput(root,c,s),query,'test',diag,dest,bs],post/'bayesian_integrated.log')
  pm=checked_manifest(dest/'manifest.json','output_hashes');assert pm['query_sha256']==sha(query) and pm['diagnostic_audit_sha256']==sha(diag)
  records.append({'model':'Bayesian IRT','kind':'bayesian','seed':'combined_posterior','status':'CONVERGED','converged':True,'probabilityfile':str(dest/'predictions.csv'),'coordinatefile':str(dest/'coordinates.csv'),'artifact_hashes':{str(dest/'manifest.json'):sha(dest/'manifest.json'),str(bayes/'manifest.json'):sha(bayes/'manifest.json')}})
 else:records.append({'model':'Bayesian IRT','kind':'bayesian','seed':'combined_posterior','status':bm.get('status'),'converged':False})
 spec={'models':records,'people_file':str(input/'people.csv'),'test_file':str(query),'external_file':str(root/'data'/c/'external_cohort.csv')}
 atomic(post/'evaluation_records.json',spec)
 if not any(r['status'] in ['COMPLETE','CONVERGED'] for r in records):
  out.mkdir(parents=True,exist_ok=True);atomic(out/'audit.json',{'status':'NO_ELIGIBLE_FITS','country':c,'size':s,'records':records,'outputs':{},'country_seal_sha256':sha(sealfile),'test_seal_hashes':{str(p):sha(p) for p in post.glob('*test_seal.json')}});return
 evaluate_condition(root,c,s,prepared,spec,out)
 audit=json.loads((out/'audit.json').read_text());audit['country_seal_sha256']=sha(sealfile);audit['test_seal_hashes']={str(p):sha(p) for p in post.glob('*test_seal.json')};atomic(out/'audit.json',audit)

def report(root):
 out=root/'results/final_report';out.mkdir(parents=True,exist_ok=True)
 jobs=json.loads((root/'jobs.json').read_text());status=[];native=[];metrics=[];external=[];contrasts=[];sources={};sensitivity=[];report_inputs={};evaluated=0
 def bind(path):
  path=Path(path);report_inputs[str(path.resolve())]=sha(path);return path
 bind(root/'jobs.json');bind(Path(__file__));bind(root/'PROTOCOL.md')
 for job in jobs['jobs']:
  if job.get('method') and job['method'] not in set(KINDS+['wnominate','bayesian']):continue
  path=root/'scheduler_state/jobs'/f'{job["id"]}.json'
  record=({'status':'REPORT_GENERATED'} if job['id']=='final_report' else json.loads(bind(path).read_text()) if path.exists() else {})
  status.append({'job':job['id'],'country':job.get('country'),'size':job.get('size'),'method':job.get('method'),'variant':job.get('variant'),'kind':job.get('kind'),'status':record.get('status','NOT_STARTED'),'returncode':record.get('returncode'),'seconds':record.get('wall_seconds'),'reason':record.get('error') or record.get('reason')})
 for c,sizes in SIZES.items():
  for s in sizes:
   p=root/'results/classic'/c/s/'base';mfile=p/'manifest.json' if (p/'manifest.json').exists() else p/'status.json'
   m=json.loads(bind(mfile).read_text()) if mfile.exists() else {'status':'NOT_STARTED'}
   row={'country':c,'requested_size':s,'status':m.get('status'),'iterations':m.get('target'),'probability_convergence':m.get('probability_convergence'),'coordinate_convergence':m.get('coordinate_convergence'),'all_item_parameter_convergence':m.get('all_item_parameter_convergence')}
   pp=prep(root,c,s)
   if (pp/'manifest.json').exists():
    pm=json.loads(bind(pp/'manifest.json').read_text());row.update(retained_people=pm['n_people'],retained_pairs=pm['n_pairs'])
   native.append(row)
   ev=root/'results/evaluation'/c/s
   if (ev/'audit.json').exists():
    audit=checked_manifest(bind(ev/'audit.json'));sources[str(ev/'audit.json')]=sha(ev/'audit.json')
    for path,h in audit.get('inputs',{}).items():assert sha(path)==h,path
    sealpath=root/'results/seals'/f'{c}.json';bind(sealpath);assert audit['country_seal_sha256']==sha(sealpath)
    for path,h in audit.get('test_seal_hashes',{}).items():assert sha(path)==h,path;bind(path)
    if audit.get('status')=='complete_condition_evaluation':
     evaluated+=1
     for name,dest in [('prediction_metrics.csv',metrics),('external_metrics.csv',external),('paired_contrasts.csv',contrasts)]:
      try:f=pd.read_csv(bind(ev/name))
      except pd.errors.EmptyDataError:continue
      f=retained_model_rows(f);f['country']=c;f['requested_size']=s;dest.append(f)
  base=root/'results/classic'/c/MID[c]/'base/validation_integrated/predictions.csv'
  for variant in ['item_shrink4','alternate_anchors']:
   p=root/'results/classic'/c/MID[c]/variant;mf=p/'manifest.json' if (p/'manifest.json').exists() else p/'status.json';m=json.loads(bind(mf).read_text()) if mf.exists() else {'status':'NOT_STARTED'}
   row={'country':c,'requested_size':MID[c],'variant':variant,'status':m.get('status')}
   alt=p/'validation_integrated/predictions.csv'
   if base.exists() and alt.exists():
    a=pd.read_csv(bind(base));b=pd.read_csv(bind(alt));z=a.merge(b,on=['source_id','person_index','pair_index','vote'],suffixes=('_base','_alternative'),validate='one_to_one');z=z[np.isfinite(z.probability_base)&np.isfinite(z.probability_alternative)]
    if len(z):
     for name in ['base','alternative']:
      pr=np.clip(z['probability_'+name].to_numpy(),1e-15,1-1e-15);yy=z.vote.to_numpy();row['validation_log_loss_'+name]=float(np.mean(-yy*np.log(pr)-(1-yy)*np.log1p(-pr)))
     row['delta_validation_log_loss']=row['validation_log_loss_alternative']-row['validation_log_loss_base'];row['sensitivity_flag_abs_delta_ge_005']=abs(row['delta_validation_log_loss'])>=.005
   sensitivity.append(row)
 pd.DataFrame(status).to_csv(out/'job_status.csv',index=False);pd.DataFrame(native).to_csv(out/'bayesian_size_diagnostics.csv',index=False);pd.DataFrame(sensitivity).to_csv(out/'prior_anchor_sensitivity.csv',index=False)
 for name,frames in [('predictive_comparison.csv',metrics),('political_selfreport_comparison.csv',external),('paired_contrasts.csv',contrasts)]:
  (pd.concat(frames,ignore_index=True) if frames else pd.DataFrame({'status':['NO_COMPLETED_EVALUATIONS']})).to_csv(out/name,index=False)
 completed=evaluated;recorded=len(sources)
 text=f'''# Comparación de tres países: estado y resultados disponibles

Evaluaciones con métricas: {completed}/10. Condiciones con estado de evaluación registrado: {recorded}/10. Este informe conserva también los trabajos fallidos, pendientes y sin convergencia suficiente. Consulte job_status.csv y bayesian_size_diagnostics.csv; una ejecución incompleta no es una derrota predictiva.

Los modelos se ajustan con exactamente las mismas personas y celdas de entrenamiento dentro de cada tamaño. El filtro20elecciones/.025 es una restricción común del diseño y no una limitación intrínseca del bayesiano. Chile mantiene su partición anterior; Francia y Brasil se reconstruyen por respuesta original, purgando pares que cruzan particiones. Las métricas entre países no identifican diferencias de polarización ni comparan geometrías entrenadas por separado.

predictive_comparison.csv contiene accuracy, logloss y Brier, con ponderación por comparación y por persona. political_selfreport_comparison.csv usa las mismas personas disponibles y folds externos, sin usar esas etiquetas para entrenar embeddings. Se distinguen promedio de resultados entre semillas y ensamble de probabilidades. El posterior bayesiano se integra sobre todos los draws de las cuatro cadenas que pasan los diagnósticos. W-NOMINATE es un ajuste nativo puntual, sin bootstrap interno del ajuste ni certificado de óptimo global.

La escala de tamaños es descriptiva y usa una muestra anidada por tamaño: no demuestra un umbral universal de convergencia. Rhat/ESS se reportan por familia. Las sensibilidades de priors y anclajes son controles en un tamaño por país; no seleccionan ganadores ni acreditan robustez de toda la muestra. Los intervalos bootstrap son puntuales y condicionales a los ajustes guardados; no llevan corrección de multiplicidad. Las pruebas chilenas son seguimiento sobre un test utilizado en análisis anteriores, no una confirmación independiente nueva.

Los tiempos fiables están en el registro del supervisor; no sumar duración de cadenas paralelas como si fuera tiempo calendario. El presupuesto de esta ejecución se define al iniciarla. El límite de iteraciones y los diagnósticos estadísticos se conservan.
'''
 (out/'report.md').write_text(text)
 for path,h in report_inputs.items():assert sha(path)==h,path
 atomic(out/'manifest.json',{'status':'AUTOMATIC_REPORT_OF_AVAILABLE_RESULTS','evaluated_conditions':completed,'recorded_conditions':recorded,'report_inputs':report_inputs,'requested_conditions':10,'generated':time.time(),'condition_audits':sources,'outputs':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'},'not_a_claim_all_models_converged':True})

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--action',choices=['seal','evaluate','report'],required=True);p.add_argument('--country');p.add_argument('--size');a=p.parse_args();root=a.root.resolve()
 if a.action=='seal':seal(root,a.country)
 elif a.action=='evaluate':evaluate(root,a.country,a.size)
 else:report(root)
if __name__=='__main__':main()

