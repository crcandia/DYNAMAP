"""Synthetic guard checks; run after test_synthetic.R and diagnostics/prediction."""
import importlib.util,json,os,subprocess,tempfile
from pathlib import Path
import pandas as pd
base=Path(__file__).resolve().parent;data=base/'test_outputs'
spec=importlib.util.spec_from_file_location('classic_diag_test',base/'diagnostics.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
snaps=[data/f'snapshot{s}' for s in [101,201,301,401]]
try:d.diagnose([snaps[0]]*4,data/'bad_duplicate_chain')
except AssertionError:pass
else:raise AssertionError('Duplicate chains accepted')
file=snaps[0]/'item_parameters.f64';saved=file.read_bytes()
try:
 file.write_bytes(bytes([saved[0]^1])+saved[1:])
 try:d.diagnose(snaps,data/'bad_tamper')
 except ValueError:pass
 else:raise AssertionError('Changed diagnostic draw accepted')
finally:file.write_bytes(saved)
assert not (data/'bad_tamper').exists()
rscript=os.environ.get('DYNAMAP_RSCRIPT','Rscript')
cmd=[rscript,str(base/'posterior_predict.R'),str(data/'prepared/prepared.rds'),str(data/'input'),str(data/'input/validation.csv'),'test',str(data/'diagnostics/audit.json'),str(data/'forbidden_test')]
result=subprocess.run(cmd,capture_output=True,text=True)
assert result.returncode!=0 and not (data/'forbidden_test').exists()
a=pd.read_csv(data/'expected.csv');b=pd.read_csv(data/'predictions/predictions.csv')
assert max(abs(a.expected-b.probability))<1e-12 and max(abs(a.plugin-b.probability))>.01
report=dict(status='PASS',synthetic_only=True,integrated_probability_maxerror=float(max(abs(a.expected-b.probability))),not_plugin=True,duplicate_chain_rejected=True,draw_tamper_rejected=True,test_without_seal_rejected=True)
(data/'guard_test_result.json').write_text(json.dumps(report,indent=2));print(report)

