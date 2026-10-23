"""Prepare one LUMIERE patient as a compressed .npz (used for the thesis; paths follow the public LUMIERE release).

    python scripts/prepare_lumiere.py /path/to/LUMIERE Patient-062 out_dir

Needs: nibabel, pandas. Writes tissue / pre / post / t1 and obs_<week> label volumes at 2 mm.
The LUMIERE data are NOT included in this repository (see data/README.md)."""
import nibabel as nib, numpy as np, json, base64, glob, os, re, pandas as pd
from scipy import ndimage as ndi
import sys
U=sys.argv[1].rstrip('/')+'/'; OUT=sys.argv[-1].rstrip('/')+'/'; os.makedirs(OUT,exist_ok=True)
def root(p): return U+p+'/' if os.path.isdir(U+p) else U+'Imaging/'+p+'/'
def pool_mean(a):
    x,y,z=[s//2*2 for s in a.shape]; a=a[:x,:y,:z].astype(np.float32)
    return a.reshape(x//2,2,y//2,2,z//2,2).mean(axis=(1,3,5))
def pool_lab(a):
    x,y,z=[s//2*2 for s in a.shape]; a=a[:x,:y,:z]
    b=a.reshape(x//2,2,y//2,2,z//2,2).transpose(0,2,4,1,3,5).reshape(x//2,y//2,z//2,8)
    cnt=np.stack([(b==c).sum(-1) for c in (1,2,3)],-1)
    return np.where(cnt.max(-1)>=3,cnt.argmax(-1)+1,0).astype(np.uint8)
def rle(a):
    f=a.ravel(order='C'); out=bytearray(); idx=np.flatnonzero(np.diff(f))+1; st=np.r_[0,idx]; en=np.r_[idx,len(f)]
    for s,e in zip(st,en):
        v=int(f[s]); L=e-s
        while L>0: c=min(L,255); out+=bytes([v,c]); L-=c
    return base64.b64encode(bytes(out)).decode()
A='/DeepBraTumIA-segmentation/atlas/'
r=pd.read_csv(U+'LUMIERE-ExpertRating-v202211.csv'); r.columns=['P','tp','lt3','nm','rate','why']
dem=pd.read_csv(U+'LUMIERE-Demographics_Pathology.csv'); dem.columns=['P','surv','sex','age','idh','idhm','mgmt','mgmtq']
c=pd.read_csv(U+'LUMIERE-datacompleteness.csv'); ok=set(zip(c[c.DeepBraTumIA=='x'].Patient,c[c.DeepBraTumIA=='x'].Timepoint))
sel=[a.replace('Patient-','') for a in sys.argv[2:-1]]
out={}; meta={'ratings':{}}
for s in sel:
    p='Patient-'+s; R=root(p); g=r[r.P==p].copy(); g['w']=g.tp.str.extract(r'week-(\d+)').astype(int)
    pre=g[g.rate=='Pre-Op'].tp.iloc[0]; post=g[g.rate=='Post-Op'].tp.iloc[0]
    fu=[(t,w,rt) for t,w,rt in zip(g.tp,g.w,g.rate) if rt not in('Pre-Op','Post-Op') and 0<w<=110 and (p,t) in ok and os.path.exists(R+t+A+'segmentation/seg_mask.nii.gz')]
    d0=R+pre+A
    bm=pool_mean(np.asanyarray(nib.load(d0+'skull_strip/brain_mask.nii.gz').dataobj))>0.5
    t1=pool_mean(np.asanyarray(nib.load(d0+'skull_strip/t1_skull_strip.nii.gz').dataobj))
    preL=pool_lab(np.asanyarray(nib.load(d0+'segmentation/seg_mask.nii.gz').dataobj))
    ii=np.argwhere(bm); lo=ii.min(0); hi=ii.max(0)+1; sl=tuple(slice(l,h) for l,h in zip(lo,hi))
    bm=bm[sl]; t1=t1[sl]; preL=preL[sl]
    v=t1[bm]; cc=np.percentile(v,[15,50,85]).astype(float)
    for _ in range(30):
        lab=np.abs(v[:,None]-cc[None]).argmin(1); cc=np.array([v[lab==k].mean() for k in range(3)])
    order=np.argsort(cc); rank=np.empty(3,int); rank[order]=np.arange(3)
    tissue=np.zeros(bm.shape,np.uint8); tissue[bm]=rank[lab]+1; tissue[(preL>0)&bm]=3
    t1u=(np.clip(t1/np.percentile(v,99.5),0,1)*255).astype(np.uint8)
    postL=pool_lab(np.asanyarray(nib.load(R+post+A+'segmentation/seg_mask.nii.gz').dataobj))[sl]
    obs={}; tags={}
    for t,w,rt in fu:
        if str(w) in obs: continue
        obs[str(w)]=pool_lab(np.asanyarray(nib.load(R+t+A+'segmentation/seg_mask.nii.gz').dataobj))[sl]; tags[w]=rt
    rec=next((w for w in sorted(tags) if tags[w]=='PD' and w>=12),None)
    core=(preL==1)|(preL==2); dist=ndi.distance_transform_edt(~core)*2
    dm=None
    if rec is not None:
        O=(obs[str(rec)]==1)|(obs[str(rec)]==2)
        if O.any(): dm=float(np.linalg.norm((np.argwhere(O).mean(0)-np.argwhere(core).mean(0))*2))
    dd=dem[dem.P==p].iloc[0]
    desc=(f'Recidivă locală (~{dm/10:.0f} cm de locul inițial)' if dm is not None and dm<=25 else f'Recidivă la distanță (~{dm/10:.0f} cm de locul inițial)') if dm is not None else ('Progresie RANO fără componentă care captează contrast' if rec is not None else 'Fără progresie RANO')
    out[p]=dict(name='Pacient '+s,desc=desc,rec=rec,sex='F' if dd.sex=='female' else 'M',age=int(dd.age),mgmt={'methylated':'metilat','not methylated':'nemetilat'}.get(dd.mgmt,'necunoscut'),
        dims=list(bm.shape),t1=base64.b64encode(t1u.tobytes()).decode(),tissue=rle(tissue),pre=rle(preL),post=rle(postL),obs={k:rle(x) for k,x in obs.items()})
    meta['ratings'][p]={('week-%03d'%w):t for w,t in tags.items()}
    np.savez_compressed(f'{OUT}{p}.npz',tissue=tissue,pre=preL,post=postL,t1=t1u,**{'obs_'+k:x for k,x in obs.items()})
    print(p,bm.shape,'rec',rec,desc,'weeks',sorted(map(int,obs)),'core cm3',round(float(core.sum())*8e-3,1),flush=True)

json.dump(meta,open(OUT+'ratings.json','w'))
