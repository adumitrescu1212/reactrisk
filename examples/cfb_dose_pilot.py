# Usage: python examples/cfb_dose_pilot.py <pacient_dir> <id> <saptamani> [t1|t2]
# Needs CFB-GBM NIfTI files (t0 gtv, brain_mask, rtdose; target-time gtv). Public: TCIA, CC BY 4.0.
import sys, numpy as np, nibabel as nib
from scipy import ndimage as ndi
sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__file__),'..'))
from reactrisk.core.operators import grid_operator
from reactrisk.core.solver import IMEXSolver
from reactrisk.core.ranking import top_k
P=sys.argv[1]; pid=int(sys.argv[2]); wk=float(sys.argv[3]); TG=sys.argv[4] if len(sys.argv)>4 else 't1'
def ld(t,n,f=1):
    d=nib.load(f'{P}/{t}/{pid}_{t}_{n}.nii.gz').get_fdata(dtype=np.float32)
    return d
def ds(a): # 0.5,0.5,1 mm -> 2 mm : block mean 4,4,2
    x,y,z=[s//k*k for s,k in zip(a.shape,(4,4,2))]
    return a[:x,:y,:z].reshape(x//4,4,y//4,4,z//2,2).mean((1,3,5))
g0=ds(ld('t0','gtv'))>0.5; g1=ds(ld(TG,'gtv'))>0.5
br=ds(ld('t0','brain_mask'))>0.5; dose=ds(ld('t0','rtdose'))
br|=g0|g1
print('vol t0 %.1f  t1 %.1f cm3'%(g0.sum()*.008,g1.sum()*.008), 'dosa max',dose.max().round(1))
D=np.where(br,0.1,0.0); A,idx=grid_operator(D,br,2.0)
rho=0.02; dt=7.0; n=int(round(wk))
def run(kill,alpha=0.15,beta=0.05,nfx=30):
    u0=np.where(g0,0.8,0.0).astype(float)
    if kill:
        d=np.clip(dose,0,None); fx=d/nfx
        S=np.exp(-(alpha*d+beta*d*fx))     # LQ survival
        u0=u0*S
    s=IMEXSolver(A,rho,dt)
    v=s.run(u0[br],n)
    out=np.zeros(br.shape); out[br]=v; return out
new=g1&~g0; k=int(new.sum())
def cov(score):
    sc=np.where(g0|~br,-1e9,score); z=np.zeros(br.shape,bool)
    o=np.argsort(sc.ravel())[::-1][:k]; z.ravel()[o]=True
    return (z&new).sum()/max(k,1)
dist=ndi.distance_transform_edt(~g0)
print('voxeli noi in t1: %d (%.1f cm3)'%(k,k*.008))
print('marja uniforma : %.3f'%cov(-dist))
print('model fara doza: %.3f'%cov(run(False)))
print('model cu doza  : %.3f'%cov(run(True)))
# doza in zona noua vs in tumora t0
print('doza medie: tumora t0 %.1f Gy | zona noua %.1f Gy | creier %.1f Gy'%(dose[g0].mean(),dose[new].mean(),dose[br].mean()))
