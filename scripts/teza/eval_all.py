import numpy as np, json, sys, glob, os
sys.path.insert(0,'.')
from fkpp import Solver
from scipy import ndimage as ndi
Vs=(20,50,100); DT=7.0; H=2.0
def load(p):
    z=np.load(p+'.npz'); obs={int(k[4:]):z[k] for k in z.files if k.startswith('obs_')}
    return z['tissue'],z['pre'],z['post'],obs
def corem(L): return (L==1)|(L==2)
def topk(score,mask,K,ties=True):
    idx=np.flatnonzero(mask.ravel()); s=score.ravel()[idx]
    o=np.argsort(-s,kind='stable'); R=np.zeros(mask.size,bool); R[idx[o[:K]]]=True; return R.reshape(mask.shape)
def rank_top(score,mask,V):
    # mask: dominio; V in cm3 -> K voxeli 8mm3; tie-aware via threshold fractional handled by mid-rank
    K=int(round(V/8e-3)); idx=np.flatnonzero(mask.ravel()); s=score.ravel()[idx]
    return topk(score,mask,K)
def cov(R,O): return float((R&O).sum()/max(O.sum(),1))
def tie_cov(score,mask,O,V):
    # acoperire cu tratare tie: voxelii cu scor egal la prag sunt impartiti proportional
    K=int(round(V/8e-3)); idx=np.flatnonzero(mask.ravel()); s=score.ravel()[idx]; o=O.ravel()[idx]
    thr=np.sort(s)[::-1][min(K,len(s))-1]; hi=s>thr; eq=s==thr; need=K-hi.sum(); frac=need/max(eq.sum(),1)
    return float((o&hi).sum()+frac*(o&eq).sum())/max(o.sum(),1)
def blur_op(mask,sig_vox):
    m=mask.astype(float); den=ndi.gaussian_filter(m,sig_vox,mode='constant')+1e-12
    def B(u):
        x=np.zeros(mask.shape); x[mask]=u; y=ndi.gaussian_filter(x,sig_vox,mode='constant')/den
        return y[mask]
    return B
def simulate(tissue,mask,u0,Dw,rho,T,jump=None):
    D=np.where(tissue==3,Dw,Dw/10).astype(float); s=Solver(D,mask,H,rho,DT,tol=1e-8); u=u0[mask].copy()
    B=blur_op(mask,jump[1]) if jump else None
    for _ in range(int(round(T/DT))):
        u=s.step(u)
        if jump:
            mu=min(1.0,jump[0]*DT); u=(1-mu)*u+mu*B(u)
    return s.to_vol(u)
def run_pair(p,start,target,res):
    tissue,pre,post,obs=load(p)
    if start=='pre': L0=pre; T0=0
    else: L0=obs[start]; T0=start
    cav=ndi.binary_dilation(corem(post) if start!='pre' else corem(pre),iterations=1)
    if start!='pre': cav=ndi.binary_dilation(corem(post),iterations=1)
    mask=(tissue>=2)&~cav
    u0=np.zeros(tissue.shape); u0[(L0==3)]=0.16; u0[corem(L0)]=0.8; u0[~mask]=0
    O=corem(obs[target])&mask
    if O.sum()==0: return
    T=(target-T0)*7.0
    startcore=corem(L0)
    dist=ndi.distance_transform_edt(~startcore)*H
    out=dict(start=str(start),target=target,Ovol=float(O.sum()*8e-3),Q={})
    # baselines
    for V in Vs:
        out['Q'][f'static{V}']=tie_cov(u0,mask,O,V)
        out['Q'][f'margin{V}']=tie_cov(-dist,mask,O,V)
    m15=(dist<=15)&mask; out['m15']=cov(m15,O); out['m15vol']=float(m15.sum()*8e-3)
    zones={}
    cfgs={'D0.03':(0.03,0.02,None),'D0.1':(0.1,0.02,None),'D0.3':(0.3,0.02,None),'J1':(0.1,0.02,(0.002,10)),'J2':(0.1,0.02,(0.006,10))}
    for k,(Dw,rho,jp) in cfgs.items():
        u=simulate(tissue,mask,u0,Dw,rho,T,jp); zones[k]=u
        for V in Vs: out['Q'][f'{k}_{V}']=tie_cov(u,mask,O,V)
    # consensus (item 8): voxel in top-V pentru >=2 din 3 valori de D
    for V in Vs:
        Z=sum(rank_top(zones[k],mask,V).astype(int) for k in ('D0.03','D0.1','D0.3'))
        out['Q'][f'cons{V}']=cov(Z>=2,O); out['Q'][f'consvol{V}']=float((Z>=2).sum()*8e-3)
    res.append(out)
if __name__=='__main__':
    pats=sys.argv[1:]
    for p in pats:
        tissue,pre,post,obs=load(p); ws=sorted(obs); meta=json.load(open('meta_all2.json'))['ratings'][p]
        pdw=[w for w in ws if meta.get('week-%03d'%w)=='PD' and w>=12]
        res=[]
        if pdw:
            run_pair(p,'pre',pdw[0],res)
            first=ws[0]; tgt=pdw[0] if first<pdw[0] else (ws[1] if len(ws)>1 else None)
            if tgt: run_pair(p,first,tgt,res)
        json.dump(res,open(f'ev_{p}.json','w')); print(p,len(res),flush=True)
