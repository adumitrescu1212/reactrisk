import numpy as np, sys
sys.path.insert(0,'.')
from fkpp import Solver
from scipy import ndimage as ndi
def setup(p):
    z=np.load(p+'.npz'); tissue=z['tissue']; pre=z['pre']
    core=(pre==1)|(pre==2); cav=ndi.binary_dilation(core,iterations=1)
    mask=(tissue>=2)&~cav
    u0=np.zeros(tissue.shape); u0[pre==3]=0.16; u0[core]=0.8
    return z,tissue,pre,core,mask,u0
def run(p,Dw,rho,week,ratio=10,dt=2.0,h=2.0):
    z,tissue,pre,core,mask,u0=setup(p)
    D=np.where(tissue==3,Dw,Dw/ratio).astype(float)
    s=Solver(D,mask,h,rho,dt)
    u=s.run(u0[mask],week*7)
    return z,core,mask,s.to_vol(u)
def cover(u,mask,obs,core,week_marg=15,h=2.0):
    O=((obs==1)|(obs==2))&mask; K=int(O.sum())
    idx=np.flatnonzero(mask.ravel()); uv=u.ravel()[idx]
    top=idx[np.argsort(-uv)[:K]]; R=np.zeros(mask.size,bool); R[top]=True; R=R.reshape(mask.shape)
    dist=ndi.distance_transform_edt(~core)*h
    base15=(dist<=15)&mask
    dv=dist.ravel()[idx]; topb=idx[np.argsort(dv)[:K]]; B=np.zeros(mask.size,bool); B[topb]=True; B=B.reshape(mask.shape)
    f=lambda M:(M&O).sum()/max(O.sum(),1)
    return K*8/1000, f(R), f(B), f(base15), base15.sum()*8/1000
if __name__=='__main__':
    for p,wk in (('Patient-062',50),('Patient-051',23)):
        print(p,wk)
        for rho in (0.01,0.02,0.04):
            for Dw in (0.05,0.2,0.8):
                z,core,mask,u=run(p,Dw,rho,wk)
                obs=z['obs_%d'%wk]
                vol,cm,cb,c15,v15=cover(u,mask,obs,core)
                print(' rho=%.2f Dw=%.2f | obs core %.1f cm3 | modelo(iso) %.2f | margine iso %.2f | margine15 %.2f (vol %.0f cm3)'%(rho,Dw,vol,cm,cb,c15,v15))
