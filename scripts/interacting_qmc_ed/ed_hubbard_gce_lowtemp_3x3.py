#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, math
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, kron, eye, diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--lx',type=int,default=3); p.add_argument('--ly',type=int,default=3)
    p.add_argument('--t',type=float,default=1.0); p.add_argument('--u',type=float,default=-5.0)
    p.add_argument('--mu',type=float,default=-1.0); p.add_argument('--beta',type=float,default=10.0)
    p.add_argument('--ph-sym-form',type=lambda x:x.lower()=='true',default=True)
    p.add_argument('--cutoff',type=float,default=1.5, help='include sectors with Emin-E0 <= cutoff')
    p.add_argument('--outdir',type=Path,required=True)
    return p.parse_args()

def gen(n,k):
    arr=[]
    for comb in combinations(range(n),k):
        s=0
        for i in comb: s|=1<<i
        arr.append(s)
    return arr

def occ(s,i): return (s>>i)&1

def cdagc(state,i,j):
    if not occ(state,j) or occ(state,i): return None
    maskj=1<<j; st=state & ~maskj
    sign1=-1 if ((state & (maskj-1)).bit_count()%2) else 1
    maski=1<<i; sign2=-1 if ((st & (maski-1)).bit_count()%2) else 1
    return sign1*sign2, st|maski

def site(x,y,lx): return x+y*lx

def hopping(lx,ly,t=1):
    n=lx*ly; rows=[]; cols=[]; vals=[]
    for y in range(ly):
      for x in range(lx):
        i=site(x,y,lx)
        for j in (site((x+1)%lx,y,lx), site(x,(y+1)%ly,lx)):
          rows += [j,i]; cols += [i,j]; vals += [-t,-t]
    return coo_matrix((vals,(rows,cols)),shape=(n,n)).tocsr()

def bilinear(n,basis,terms):
    idx={s:i for i,s in enumerate(basis)}; rows=[]; cols=[]; vals=[]
    for col,state in enumerate(basis):
      for i,j,amp in terms:
        r=cdagc(state,i,j)
        if r:
          sign,ns=r; rows.append(idx[ns]); cols.append(col); vals.append(amp*sign)
    return coo_matrix((vals,(rows,cols)),shape=(len(basis),len(basis))).tocsr()

def onespin(tmat,basis):
    coo=tmat.tocoo(); return bilinear(tmat.shape[0], basis, [(int(i),int(j),float(a)) for i,j,a in zip(coo.row,coo.col,coo.data)])

def x_terms(lx,ly,t,qx=0,qy=0,current=True):
    terms=[]
    for y in range(ly):
      for x in range(lx):
        src=site(x,y,lx); dst=site((x+1)%lx,y,lx); phase=np.exp(1j*(qx*x+qy*y))
        if current:
          terms += [(dst,src,1j*t*phase),(src,dst,-1j*t*phase)]
        else:
          terms += [(dst,src,-t+0j),(src,dst,-t+0j)]
    return terms

def build_sector(args,nup,ndn):
    n=args.lx*args.ly; up=gen(n,nup); dn=gen(n,ndn); du=len(up); dd=len(dn)
    tmat=hopping(args.lx,args.ly,args.t)
    hup=onespin(tmat,up); hdn=onespin(tmat,dn)
    H=kron(eye(dd,format='csr'),hup)+kron(hdn,eye(du,format='csr'))
    const=(-0.5*args.u*(nup+ndn)+0.25*args.u*n-args.mu*(nup+ndn)) if args.ph_sym_form else (-args.mu*(nup+ndn))
    docc=[]
    for d in dn:
      for u in up: docc.append((u&d).bit_count())
    H=H+diags([args.u*x+const for x in docc])
    return H.tocsr(), up, dn, np.array(docc,float)

def emin(args,nup,ndn):
    H,_,_,_=build_sector(args,nup,ndn)
    if H.shape[0]==1: return float(H[0,0]), H.shape[0]
    return float(eigsh(H,k=1,which='SA',return_eigenvectors=False,tol=1e-10)[0]), H.shape[0]

def static_response(evals,Aeig,beta,e0):
    e=evals; de=e[:,None]-e[None,:]
    # kernel with m=row, n=col: (exp(-β En)-exp(-β Em))/(Em-En)
    wm=np.exp(-beta*(e-e0))
    kern=np.empty_like(de)
    mask=np.abs(de)<1e-10
    kern[mask]=np.broadcast_to(beta*wm[:,None], de.shape)[mask]
    kern[~mask]=(wm[None,:]-wm[:,None])[~mask]/de[~mask]
    return float(np.sum(np.abs(Aeig)**2 * kern).real)

def main():
    args=parse_args(); n=args.lx*args.ly; beta=args.beta
    bounds=[]
    for nup in range(n+1):
      for ndn in range(n+1):
        e,d=emin(args,nup,ndn); bounds.append((e,nup,ndn,d))
    bounds.sort(); e0=bounds[0][0]
    selected=[b for b in bounds if b[0]-e0 <= args.cutoff]
    print('E0',e0,'selected sectors',selected)
    all_e=[]; all_N=[]; all_D=[]
    Zs=0; Es=Ns=N2s=Ds=Kxs=LLs=LTs=0.0
    qmin=2*math.pi/args.lx
    sector_rows=[]
    for e_min,nup,ndn,dim in selected:
      H,up,dn,docc=build_sector(args,nup,ndn)
      Hd=H.toarray()
      evals, vecs=eigh(Hd, check_finite=False)
      w=np.exp(-beta*(evals-e0)); Zs += np.sum(w); Es += np.dot(w,evals); Ns += np.sum(w)*(nup+ndn); N2s += np.sum(w)*(nup+ndn)**2
      docc_eig=np.sum(np.abs(vecs)**2 * docc[:,None], axis=0); Ds += np.dot(w,docc_eig)
      du=len(up); dd=len(dn); Iu=eye(du,format='csr'); Id=eye(dd,format='csr')
      jL=kron(Id,bilinear(n,up,x_terms(args.lx,args.ly,args.t,qx=qmin,current=True)))+kron(bilinear(n,dn,x_terms(args.lx,args.ly,args.t,qx=qmin,current=True)),Iu)
      jT=kron(Id,bilinear(n,up,x_terms(args.lx,args.ly,args.t,qy=qmin,current=True)))+kron(bilinear(n,dn,x_terms(args.lx,args.ly,args.t,qy=qmin,current=True)),Iu)
      kx=kron(Id,bilinear(n,up,x_terms(args.lx,args.ly,args.t,current=False)))+kron(bilinear(n,dn,x_terms(args.lx,args.ly,args.t,current=False)),Iu)
      JL=vecs.conj().T @ jL.toarray() @ vecs; JT=vecs.conj().T @ jT.toarray() @ vecs; KX=vecs.conj().T @ kx.toarray() @ vecs
      LLs += static_response(evals,JL,beta,e0); LTs += static_response(evals,JT,beta,e0); Kxs += float(np.dot(w,np.real(np.diag(KX))))
      sector_rows.append((nup,ndn,dim,e_min,evals[-1]))
    E=Es/Zs; N=Ns/Zs; N2=N2s/Zs; D=Ds/Zs
    if args.ph_sym_form:
      interaction=args.u*(D-0.5*N+0.25*n); muE=-args.mu*N
    else:
      interaction=args.u*D; muE=-args.mu*N
    kinetic=E-interaction-muE
    LL=LLs/(Zs*n); LT=LTs/(Zs*n); KX=Kxs/(Zs*n); rho=0.25*(LL-LT); rhod=0.25*(-KX-LT)
    comp=beta/n*(N2-N*N)
    args.outdir.mkdir(parents=True,exist_ok=True)
    with open(args.outdir/'summary.tsv','w',newline='') as f:
      cols=['beta','mu','t','U','ph_sym_form','sector_cutoff','selected_sector_count','logZ_shifted','total_energy','energy_per_site','kinetic_energy','interaction_energy','chemical_potential_energy','N','density','double_occupancy','double_occupancy_per_site','compressibility','lambda_longitudinal_qmin0','lambda_transverse_0qmin','rho_s_current','Kx_per_site','diamagnetic_minus_Kx_per_site','rho_s_diamagnetic']
      wr=csv.DictWriter(f,fieldnames=cols,delimiter='\t'); wr.writeheader(); wr.writerow(dict(zip(cols,[beta,args.mu,args.t,args.u,args.ph_sym_form,args.cutoff,len(selected),math.log(Zs)-beta*e0,E,E/n,kinetic,interaction,muE,N,N/n,D,D/n,comp,LL,LT,rho,KX,-KX,rhod])))
    with open(args.outdir/'selected_sectors.tsv','w',newline='') as f:
      wr=csv.writer(f,delimiter='\t'); wr.writerow(['nup','ndn','dimension','min_eigenvalue','max_eigenvalue']); wr.writerows(sector_rows)
    print('E/site',E/n,'density',N/n,'docc/site',D/n,'LL',LL,'LT',LT,'rho',rho,'-Kx/N',-KX,'rhod',rhod,'out',args.outdir)
if __name__=='__main__': main()
