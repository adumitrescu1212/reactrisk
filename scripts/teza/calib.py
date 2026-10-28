import numpy as np, sys, json, itertools
sys.path.insert(0,'.')
from ref_sim import setup
from fkpp import Solver
Vs=(20,50,100)
def cov_at(u,mask,O,V):
    idx=np.flatnonzero(mask.ravel()); uv=u.ravel()[idx]; k=int(V/8e-3)
    top=idx[np.argpartition(-uv,k)[:k]]; R=np.zeros(mask.size,bool); R[top]=True
    return float((R.reshape(mask.shape)&O).sum()/max(O.sum(),1))
cfg={'Patient-062':dict(weeks=[15,30,42,50,57,64]),'Patient-051':dict(weeks=[12,23,36,50,64])}
Ds=[0.01,0.03,0.1,0.3,1.0,3.0]; Rs=[0.01,0.02,0.04]; dt=3.5
res={}
for p,c in cfg.items():
    z,tissue,pre,core,mask,u0=setup(p)
    obs={w:((z['obs_%d'%w]==1)|(z['obs_%d'%w]==2))&mask for w in c['weeks']}
    res[p]={}
    for Dw,rho in itertools.product(Ds,Rs):
        D=np.where(tissue==3,Dw,Dw/10).astype(float); s=Solver(D,mask,2.0,rho,dt); u=u0[mask].copy(); out={}
        steps=int(round(max(c['weeks'])*7/dt)); wk_at={int(round(w*7/dt)):w for w in c['weeks']}
        for n in range(1,steps+1):
            u=s.step(u)
            if n in wk_at:
                w=wk_at[n]; v=s.to_vol(u); out[w]=[cov_at(v,mask,obs[w],V) for V in Vs]
        res[p]['%g_%g'%(Dw,rho)]=out; print(p,Dw,rho,{w:[round(x,2) for x in o] for w,o in out.items()},flush=True)
    json.dump(res,open('calib.json','w'))
