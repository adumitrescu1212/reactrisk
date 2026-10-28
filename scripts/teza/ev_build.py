import json,glob,re,numpy as np
from scipy import stats
desc={}
for f in ('export_new.log',):
    for l in open(f):
        m=re.match(r'Patient-(\d+) .*? rec (\S+) (.*?) weeks',l)
        if m: desc[m.group(1)]=m.group(3)
old={r['p']:r['desc'] for r in json.load(open('ev_summary.json'))}; desc.update(old)
rows=[]
for f in sorted(glob.glob('ev_Patient-*.json')):
    p=re.search(r'Patient-(\d+)',f).group(1)
    for x in json.load(open(f)):
        Q=x['Q']; g=lambda k:[round(Q[f'{k}_{V}'],3) for V in (20,50,100)]
        t0=0 if x['start']=='pre' else int(x['start'])
        rows.append(dict(p=p,desc=desc.get(p,''),s='pre' if x['start']=='pre' else 'ctrl',t0=t0,t=x['target'],O=round(x['Ovol'],1),
          m15=round(x['m15'],3),v15=round(x['m15vol']),model=g('D0.1'),jump=g('J1'),stat=[round(Q[f'static{V}'],3) for V in (20,50,100)],
          marg=[round(Q[f'margin{V}'],3) for V in (20,50,100)],cons=[round(Q[f'cons{V}'],3) for V in (20,50,100)]))
json.dump(rows,open('ev_summary.json','w'))
rng=np.random.default_rng(1); S={}
for s in ('pre','ctrl'):
    R=[r for r in rows if r['s']==s and r['O']>=1]; S[s]={'n':len(R)}
    for vi,V in enumerate((20,50,100)):
        for a,b in (('model','marg'),('model','stat'),('jump','marg')):
            d=np.array([r[a][vi]-r[b][vi] for r in R]); bs=[rng.choice(d,len(d)).mean() for _ in range(5000)]
            try: pw=float(stats.wilcoxon(d).pvalue) if np.any(d!=0) else 1.0
            except Exception: pw=1.0
            S[s][f'{a}-{b}_{V}']=[round(float(d.mean()),3),round(float(np.percentile(bs,2.5)),3),round(float(np.percentile(bs,97.5)),3),round(pw,3),int((d>0).sum()),int((d<0).sum())]
json.dump(S,open('ev_stats.json','w')); print(json.dumps(S,indent=0)[:3000]); print(len(rows),'rows')
