"""Matched-support evaluation. Reads saved predictions only, never trains preference models.

Records JSON: {models:[{model, seed, status, probabilityfile, coordinatefile,
 optional user_indices, optional artifact_hashes}], test_file, people_file,
 external_file}. Status must explicitly be complete/converged/success. Bayesian
 records additionally require converged=true. Files may be absolute or root-relative.
 Native coordinates: person_index,x1,x2 (local 1-based). Neural NPZ: users and
 user_indices (global dense IDs), or record.user_indices supplies the latter.

Primary predictive aggregation averages per-person metrics across optimizer seeds.
 Probability ensembles are separately labelled. Chains must already be combined
 into ONE posterior record, never passed as optimizer replicas. Bootstrap intervals
 are pointwise and conditional on saved fits and the common evaluated people.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

SEED=20261013
N_BOOT=1000
ACTIVE_MODELS=frozenset(['B1','B2','B3','B4','W-NOMINATE','Bayesian IRT'])

def retained_model_rows(frame):
    """Keep the active scope when collecting immutable earlier evaluations."""
    keep=pd.Series(True,index=frame.index)
    for column in ('model','model_a','model_b'):
        if column in frame: keep &= frame[column].isin(ACTIVE_MODELS)
    return frame.loc[keep].copy()

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def resolve(root,value):
    p=Path(value)
    return p if p.is_absolute() else Path(root)/p

def metrics(y,p):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float)
    if not (np.isfinite(p).all() and np.isin(y,[0,1]).all() and ((p>=0)&(p<=1)).all()):
        raise ValueError('Invalid binary outcomes or probabilities')
    q=np.clip(p,1e-15,1-1e-15)
    return {'log_loss':-(y*np.log(q)+(1-y)*np.log1p(-q)),
            'brier':(p-y)**2,
            'accuracy':np.where(p==.5,.5,((p>.5)==y).astype(float))}

def paired_interval(d):
    d=np.asarray(d,dtype=float)
    if len(d)==0 or not np.isfinite(d).all(): raise ValueError('Invalid paired contrasts')
    rng=np.random.default_rng(SEED)
    # Bounded memory, one resampled participant is the uncertainty unit.
    samples=np.empty(N_BOOT)
    for i in range(N_BOOT): samples[i]=d[rng.integers(0,len(d),len(d))].mean()
    lo,hi=np.quantile(samples,[.025,.975])
    return float(d.mean()),float(lo),float(hi)

def check_keys(df,cols,name):
    if df[cols].isna().any().any() or df.duplicated(cols).any():
        raise ValueError(f'{name}: missing/duplicate keys {cols}')

def read_coordinates(root,record,people,read):
    p=resolve(root,record['coordinatefile']); read[p]=sha(p)
    if p.suffix=='.npz':
        with np.load(p,allow_pickle=False) as z:
            x=np.asarray(z['users'],float)
            ids=np.asarray(z['user_indices']) if 'user_indices' in z else record.get('user_indices')
        if isinstance(ids,str):
            ip=resolve(root,ids); read[ip]=sha(ip); ids=np.load(ip,allow_pickle=False)
        if ids is None: raise ValueError('Neural coordinates require explicit global user_indices')
        ids=np.asarray(ids)
    else:
        f=pd.read_csv(p)
        cols=record.get('coordinate_columns',['x1','x2'])
        if 'person_index' not in f or not set(cols)<=set(f):
            raise ValueError('Native coordinates need person_index and explicit coordinate_columns')
        check_keys(f,['person_index'],'coordinates')
        f=f.merge(people[['person_index','global_user_index']],on='person_index',validate='one_to_one',how='left')
        ids=f.global_user_index.to_numpy(); x=f[cols].to_numpy(float)
    if x.ndim!=2 or x.shape[1]!=2 or len(ids)!=len(x): raise ValueError('Primary classifier requires exactly two coordinates')
    if not np.isfinite(x).all() or not np.isfinite(np.asarray(ids,float)).all(): raise ValueError('Nonfinite coordinates/IDs')
    out=pd.DataFrame({'user_index':ids,'x1':x[:,0],'x2':x[:,1]})
    check_keys(out,['user_index'],'coordinate global IDs')
    if not set(out.user_index)<=set(people.global_user_index): raise ValueError('Unknown coordinate user')
    return out

def evaluate_condition(root,country,size,prepared_dir,model_records,out):
    root=Path(root); prepared_dir=Path(prepared_dir); out=Path(out)
    spec=model_records if isinstance(model_records,dict) else {'models':model_records}
    if any(r['model'] not in ACTIVE_MODELS for r in spec['models']):
        raise ValueError('Evaluation records contain methods outside the active benchmark scope')
    read={}
    def csv(path):
        p=resolve(root,path); read[p]=sha(p); return pd.read_csv(p)
    people_path=spec.get('people_file',str(prepared_dir/'people.csv'))
    test_path=spec.get('test_file',str(root/'evaluation'/country/str(size)/'test.csv'))
    external_path=spec.get('external_file',str(root/'data'/country/'external_cohort.csv'))
    people=csv(people_path); test=csv(test_path)
    check_keys(people,['person_index'],'people'); check_keys(people,['global_user_index'],'global people')
    check_keys(test,['source_id'],'test')
    if not set(test.person_index)<=set(people.person_index): raise ValueError('Unknown test person')
    if not np.isin(test.vote,[0,1]).all(): raise ValueError('Test vote must be binary 0/1')
    selected=[]; excluded=[]; predictions={}; coordinates={}; inventory=[]
    for r in spec['models']:
        rid=f"{r['model']}__{r['seed']}"
        if rid in predictions or any(x.get('id')==rid for x in excluded): raise ValueError('Duplicate model/seed record')
        native_bayes=r.get('kind')=='bayesian' or any(s in r['model'].lower() for s in ('pscl','bayes','ideal'))
        if str(r.get('status','')).lower() not in ('complete','converged','success') or (native_bayes and r.get('converged') is not True):
            excluded.append({'id':rid,'reason':'not explicitly successful/converged','record':r}); continue
        for name,digest in r.get('artifact_hashes',{}).items():
            p=resolve(root,name)
            if sha(p)!=digest: raise ValueError(f'Artifact changed: {p}')
            read[p]=digest
        p=csv(r['probabilityfile'])
        check_keys(p,['source_id'],rid)
        if not {'vote','person_index','probability'}<=set(p): raise ValueError('Prediction CSV schema mismatch')
        ref=test[['source_id','vote','person_index']].merge(p[['source_id','vote','person_index','probability']+(['covered'] if 'covered' in p else [])],on='source_id',how='inner',suffixes=('_true',''),validate='one_to_one')
        if len(ref)!=len(p) or not (ref.vote_true.eq(ref.vote).all() and ref.person_index_true.eq(ref.person_index).all()):
            raise ValueError(f'Target/ID mismatch: {rid}')
        input_rows=len(ref)
        if 'covered' in ref:
            if not ref.covered.isin([True,False,0,1,'TRUE','FALSE','true','false']).all(): raise ValueError('Invalid coverage flag')
            ref=ref[ref.covered.isin([True,1,'TRUE','true'])].copy()
        metrics(ref.vote,ref.probability)
        predictions[rid]=ref[['source_id','person_index','vote','probability']].set_index('source_id')
        if r.get('coordinatefile'): coordinates[rid]=read_coordinates(root,r,people,read)
        selected.append((rid,r))
        inventory.append({'record':rid,'model':r['model'],'seed':r['seed'],'rows':len(ref),'input_rows':input_rows,'people':ref.person_index.nunique(),'row_coverage':len(ref)/len(test) if len(test) else None})
    if not selected: raise ValueError('No scientifically eligible model records')
    common=set(test.source_id)
    for p in predictions.values(): common &= set(p.index)
    ids=test.loc[test.source_id.isin(common),'source_id'].to_numpy()
    if len(ids)==0: raise ValueError('Empty common prediction support')
    rows=[]; perperson=[]; model_probs={}
    for rid,r in selected:
        p=predictions[rid].loc[ids]; m=metrics(p.vote,p.probability)
        f=pd.DataFrame(m); f['person_index']=p.person_index.to_numpy()
        g=f.groupby('person_index',sort=True).mean(); g['model']=r['model']; g['record']=rid
        perperson.append(g.reset_index())
        for metric,values in m.items():
            rows.append({'model':r['model'],'record':rid,'aggregation':'individual_fit','metric':metric,'micro':float(values.mean()),'macro':float(g[metric].mean()),'people':len(g),'rows':len(p)})
        model_probs.setdefault(r['model'],[]).append(p.probability.to_numpy())
    person=pd.concat(perperson,ignore_index=True)
    primary=person.groupby(['model','person_index'],sort=True)[['log_loss','brier','accuracy']].mean().reset_index()
    for model in model_probs:
        rr=pd.DataFrame(rows); rr=rr[(rr.model==model)&(rr.aggregation=='individual_fit')]
        for metric in ('log_loss','brier','accuracy'):
            t=rr[rr.metric==metric]
            rows.append({'model':model,'record':'seed_average','aggregation':'mean_fit_metric','metric':metric,'micro':float(t.micro.mean()),'macro':float(t.macro.mean()),'seed_sd_micro':float(t.micro.std(ddof=1)) if len(t)>1 else None,'seed_sd_macro':float(t.macro.std(ddof=1)) if len(t)>1 else None,'fits':len(t),'people':int(t.people.iloc[0]),'rows':len(ids)})
        base=predictions[next(rid for rid,r in selected if r['model']==model)].loc[ids]
        for metric,values in metrics(base.vote,np.mean(model_probs[model],axis=0)).items():
            macro=pd.DataFrame({'u':base.person_index.to_numpy(),'v':values}).groupby('u').v.mean().mean()
            rows.append({'model':model,'record':'probability_ensemble','aggregation':'probability_ensemble','metric':metric,'micro':float(values.mean()),'macro':float(macro),'fits':len(model_probs[model]),'people':base.person_index.nunique(),'rows':len(ids)})
    contrasts=[]
    for a,b in itertools.combinations(sorted(model_probs),2):
        pair=primary[primary.model==a].merge(primary[primary.model==b],on='person_index',suffixes=('_a','_b'),validate='one_to_one')
        for metric in ('log_loss','brier','accuracy'):
            delta,lo,hi=paired_interval(pair[metric+'_a']-pair[metric+'_b'])
            contrasts.append({'model_a':a,'model_b':b,'metric':metric,'estimand':'person_macro_mean_fit_metric','a_minus_b':delta,'ci95_low':lo,'ci95_high':hi,'people':len(pair),'multiplicity':'pointwise, no familywise superiority decision'})
    external_rows=[]; oof=[]; ext_status='not_available'
    if coordinates:
        ext=csv(external_path); check_keys(ext,['user_index'],'external')
        if not np.isin(ext.label,[0,1]).all() or ext.fold.isna().any(): raise ValueError('Invalid external labels/folds')
        common_ext=set(ext.user_index)
        for c in coordinates.values(): common_ext &= set(c.user_index)
        ext=ext[ext.user_index.isin(common_ext)].sort_values('user_index').reset_index(drop=True)
        if len(ext) and ext.fold.nunique()>=2:
            ext_status='evaluated_common_coordinate_support'
            for rid,c in coordinates.items():
                f=ext.merge(c,on='user_index',validate='one_to_one'); prob=np.full(len(f),np.nan)
                for fold in sorted(f.fold.unique()):
                    tr=f.fold.ne(fold).to_numpy(); te=~tr
                    if f.loc[tr,'label'].nunique()!=2: raise ValueError('External fold training lacks both classes')
                    clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=42))
                    clf.fit(f.loc[tr,['x1','x2']],f.loc[tr,'label']); prob[te]=clf.predict_proba(f.loc[te,['x1','x2']])[:,1]
                    if clf[-1].n_iter_.max()>=2000: raise ValueError('External classifier failed convergence')
                mm=metrics(f.label,prob); model=next(r['model'] for rid2,r in selected if rid2==rid)
                external_rows.append({'model':model,'record':rid,'people':len(f),'accuracy':float(mm['accuracy'].mean()),'log_loss':float(mm['log_loss'].mean()),'brier':float(mm['brier'].mean()),'auc':float(roc_auc_score(f.label,prob)) if f.label.nunique()==2 else None,'balanced_accuracy':float(balanced_accuracy_score(f.label,prob>.5))})
                oo=f[['user_index','label','fold']].copy(); oo['probability']=prob; oo['record']=rid; oo['model']=model; oof.append(oo)
        else: ext_status='not_estimable_empty_or_one_fold_common_support'
    # Bind every input again before publishing. Existing output directory is rejected.
    for p,digest in read.items():
        if sha(p)!=digest: raise ValueError(f'Input changed during evaluation: {p}')
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise ValueError('Refusing to overwrite existing evaluation outputs')
    pd.DataFrame(rows).to_csv(out/'prediction_metrics.csv',index=False)
    person.to_csv(out/'person_metrics.csv',index=False)
    primary.to_csv(out/'mean_fit_person_metrics.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'paired_contrasts.csv',index=False)
    pd.DataFrame(inventory).to_csv(out/'coverage.csv',index=False)
    pd.DataFrame(external_rows).to_csv(out/'external_metrics.csv',index=False)
    if oof: pd.concat(oof,ignore_index=True).to_csv(out/'external_oof.csv',index=False)
    note=f"# {country}, requested size {size}\n\nCommon predictive support: {len(ids)} of {len(test)} test rows. Primary metrics average each participant's metric across fitted replicas. Probability ensembles are secondary and explicitly labelled. Exact probability ties receive half accuracy credit. Log loss clips probabilities at 1e-15. Intervals are 1,000 participant bootstraps conditional on saved fits, pointwise without multiplicity adjustment.\n\nExternal status: {ext_status}. Classifiers use the same available participant intersection and supplied fixed folds, training-only standardization and C=1. MF's raw two-dimensional coordinates are an explicitly representation-dependent linear-classifier comparison, not proof against alternative MF features. Failed or unconverged records are excluded, never assigned losing scores. No cross-country geometric comparisons or convergence thresholds are inferred.\n"
    (out/'report.md').write_text(note)
    audit={'country':country,'size':str(size),'status':'complete_condition_evaluation','excluded':excluded,'records':[{k:v for k,v in r.items()} for _,r in selected],'external_status':ext_status,'common_rows':len(ids),'total_test_rows':len(test),'source_sha256':sha(__file__),'inputs':{str(p):d for p,d in read.items()},'outputs':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'bootstrap_seed':SEED,'bootstrap_draws':N_BOOT,'scope':'conditional paired comparisons, no global completion gate'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    return audit

def collect_conditions(root,out):
    root=Path(root); out=Path(out); tables=[]; sources={}
    for p in sorted(root.glob('**/audit.json')):
        try: a=json.loads(p.read_text())
        except (ValueError,OSError): continue
        if a.get('status')!='complete_condition_evaluation': continue
        for name,digest in a['outputs'].items():
            if sha(p.parent/name)!=digest: raise ValueError(f'Changed condition output {p.parent/name}')
        for name,digest in a['inputs'].items():
            if sha(name)!=digest: raise ValueError(f'Changed condition input {name}')
        f=retained_model_rows(pd.read_csv(p.parent/'prediction_metrics.csv'))
        if f.empty: continue
        f['country']=a['country']; f['size']=a['size']; tables.append(f); sources[str(p)]=sha(p)
    if not tables: raise ValueError('No completed condition evaluations')
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise ValueError('Refusing to overwrite collector')
    pd.concat(tables,ignore_index=True).to_csv(out/'condition_metrics.csv',index=False)
    (out/'manifest.json').write_text(json.dumps({'condition_audits':sources,'active_models':sorted(ACTIVE_MODELS),'status':'available_conditions_only_not_full_completion','output_sha256':sha(out/'condition_metrics.csv')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--country'); p.add_argument('--size'); p.add_argument('--prepared'); p.add_argument('--records'); p.add_argument('--out',required=True); p.add_argument('--collect',action='store_true'); a=p.parse_args()
    if a.collect: collect_conditions(a.root,a.out)
    else:
        if not all((a.country,a.size,a.prepared,a.records)): p.error('Condition evaluation needs country,size,prepared,records')
        evaluate_condition(a.root,a.country,a.size,a.prepared,json.loads(Path(a.records).read_text()),a.out)

