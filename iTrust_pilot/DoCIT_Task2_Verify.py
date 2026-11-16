"""Check the recorded DoCIT Task1 ZIP and reproduce exploratory MAP comparison.
Run: python DoCIT_Task2_Verify.py path/to/DoCIT_iTrust_Task1_results.zip
Dependencies: pandas, numpy, scipy.
"""
import sys, zipfile, io, re
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, spearmanr
path=sys.argv[1] if len(sys.argv)>1 else 'DoCIT_iTrust_Task1_results.zip'
with zipfile.ZipFile(path) as z:
    per=pd.read_csv(io.BytesIO(z.read('per_requirement_metrics.csv')))
    ranking=pd.read_csv(io.BytesIO(z.read('ranked_candidates.csv')))
    chunks=pd.read_csv(io.BytesIO(z.read('code_chunk_diagnostics.csv')))
assert len(ranking)==142380 and len(per)==630
for (model,q),g in ranking.groupby(['Model','Requirement']):
    assert len(g)==226 and g.Rank.nunique()==226 and g.JavaFile.nunique()==226
    pos=np.flatnonzero(g.sort_values('Rank').RecordedLink.to_numpy()==1)+1
    expected=np.mean(np.arange(1,len(pos)+1)/pos)
    actual=per[(per.Model==model)&(per.Requirement==q)]['AP'].iloc[0]
    assert abs(expected-actual)<2e-8
w=per.pivot(index='Requirement',columns='Model',values='AP')
b='UniXcoder B | cleaned, all windows'; tf='TF-IDF'
delta=w[b]-w[tf]
print('TF-IDF MAP:',w[tf].mean())
print('UniXcoder B MAP:',w[b].mean())
print('MAP difference:',delta.mean(), 'Wilcoxon p:',wilcoxon(delta).pvalue)
print('B improved/worsened/same:',sum(delta>1e-8),sum(delta< -1e-8),sum(abs(delta)<=1e-8))
uc=w.index.to_series().str.extract(r'^(UC\d+)')[0]
clusters=[group.to_numpy() for _,group in pd.DataFrame({'uc':uc,'delta':delta}).groupby('uc').delta]
rng=np.random.default_rng(635)
boot=[]
for i in range(10000):
    idx=rng.integers(0,len(clusters),len(clusters))
    boot.append(np.concatenate([clusters[j] for j in idx]).mean())
print('95% use-case group bootstrap CI:',np.quantile(boot,[.025,.975]))
