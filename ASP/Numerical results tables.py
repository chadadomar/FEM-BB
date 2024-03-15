#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from mesh_curl_2D import *
from Mass_curl_2d import *
from Stiffness_curl_2d import *
from Loadvector_curl_2d import *
from Evaluation_curl_BBform import *
from Quadratur_over_triangle import *

import numpy as np
import time
import os

from numpy import linalg as la
from scipy.sparse.linalg import cg, gmres, bicgstab, minres
from tabulate import tabulate


sprint = lambda x: '{:.2e}'.format(x)

np.seterr('raise')



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                   Data
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

ps            = (1,2,3,4,5,6,7,8)
ks            = (7,8) #9,10,11,12,13
ncells        = {2:"4", 3:"8", 4:"16", 5:"44", 6:"101", 7:"215", 8:"401", 9:"800", 10:"1586", 11:"3199", 12:"6354", 13:"12770", 14:"25497", 15:"50917", 16:"101741", 17:"203504", 18:"406760"}
Taus          = [10**k for k in range(-4,5)]

# Right hand side and solution
def f(x,y):
    return np.array([y*(1-y)+2,x*(1-x)+2])

def uf(x,y):
    return np.array([y*(1-y),x*(1-x)])

def curluf(x,y):
    return 2*(y-x)

def curlcurluf(x,y):
    return np.array([2,2])



def uh(x,y):
    return np.array([np.sin(np.pi*y),np.sin(np.pi*x)])

def curlcurluh(x,y):
    return np.array( [  np.pi**2*np.sin(np.pi*y), np.pi**2*np.sin(np.pi*x) ] )

def curluh(x,y):
    return np.pi*np.cos(np.pi*x) - np.pi*np.cos(np.pi*y)


def scd(x,y):
    return np.array([1,1])

def c1(tau):
    return  -1/(2* tau * np.cosh( np.sqrt(tau)/2 ))

def utau(tau,x,y):
    return c1(tau) * np.array([ 2* np.cosh( np.sqrt(tau)*(y-0.5) ) , 2* np.cosh( np.sqrt(tau)*(x-0.5) ) ])  + 1/tau * np.array([1,1])

def curlutau(tau,x,y):
    return 2* (np.sqrt(tau)) * c1(tau) * (np.sinh( np.sqrt(tau)*(x-0.5)) - np.sinh( np.sqrt(tau)*(y-0.5)) )

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                    Creates files which contains results
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def mkdir_p(dir):
    # type: (unicode) -> None
    if os.path.isdir(dir):
        return
    os.makedirs(dir)

def create_folder(tau,p):

    tau="tau "+str(tau)
    p  ="p "+str(p)

    folder = 'numerical_results/{tau}/{p}'.format(tau=tau,p=p)

    mkdir_p(os.path.join(folder, 'txt'))
    mkdir_p(os.path.join(folder, 'tex'))

    return folder


def write_table(d, folder, kind):
    #headers = ['grid/degree p']

    #for p in ps:
        #headers.append(str(p))

    # add table rows
    rows = []
    for k in ks:
        ncell = ncells[k]
        row = ['$' + ncell +  '$']
        value = d[k][kind]
        if isinstance(value, str):
            v = value
        elif isinstance(value, int):
            v = '$'+str(value) +'$'
        else:
            v =  '$'+sprint(value)+'$'
        row.append(v)
        rows.append(row)

    table = tabulate(rows)

    fname = '{label}.txt'.format(label=kind)
    fname = os.path.join('txt', fname)
    fname = os.path.join(folder, fname)

    with open(fname, 'w', encoding="utf-8") as f:
        table = tabulate(rows, tablefmt ='fancy_grid')
        f.write(str(table))

    fname = '{label}.tex'.format(label=kind)
    fname = os.path.join('tex', fname)
    fname = os.path.join(folder, fname)

    with open(fname, 'w') as f:
        table = tabulate(rows, tablefmt='latex_raw')
        f.write(str(table))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                    The main function
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def reconstruct(X,I):
    # creat new vector with newX:
        # lenth = sum of lengths of X and I
        # newX[i]=0 for i in I
    li=len(I)
    n=len(X)+li
    newX=np.zeros(n)
    x=0
    i=0
    flag=True
    for k in range(n):
        if flag and k==I[i]:
            i+=1
            if i==li:
                flag=False
        else:
            newX[k]=X[x]
            x+=1
    return newX


def main(k, p, tau):
    r=p
    print('============tau = {tau}, p = {p}, k = {k}============'
          .format(tau = tau, p=p,k=k))
    
    tb = time.time()

    mesh_points,mesh_tris,mesh_edges,tris_edges=mesh(k)
    ntris=len(mesh_tris)
    nedges=len(mesh_edges)


    ndof=nbr_globDof(nedges,ntris,r)
    M=np.zeros((ndof,ndof))
    S=np.zeros((ndof,ndof))
    F=np.zeros(ndof)

    #def rh(x,y):
        #return curlcurluh(x,y)+ tau* uh(x,y)

    def sol_exct(x,y):
        return utau(tau,x,y)
    def curl_sol_exct(x,y):
        return curlutau(tau,x,y)

    for i in range(ntris):
        T=mesh_tris[i]

        # vertex of the triangle/elem tr
        p0=mesh_points[T[0]]
        p1=mesh_points[T[1]]
        p2=mesh_points[T[2]]

        # liste of vertices's coordinates
        Liste=[p0[0],p0[1],p1[0],p1[1], p2[0],p2[1]]

        # Local element
        St=Stiff2d(Liste,r)
        Mt=mass2d(Liste,r)
        Ft=load2d(scd,Liste,r)

        local_ndof=r*(r+2)
        for j in range(local_ndof):
            glob_j,sign_j=local_to_global(nedges,T, tris_edges[i], i ,j,r)
            F[glob_j]+= sign_j * Ft[j]
            for k in range(local_ndof):
                glob_k,sign_k=local_to_global(nedges,T, tris_edges[i], i ,k,r)
                S[glob_j][glob_k]+= sign_j * sign_k * St[j][k]
                M[glob_j][glob_k]+= sign_j * sign_k* Mt[j][k]

    I=IndexToDelete(mesh_edges,mesh_points,r)
    S=np.delete(S, I,0)
    S=np.delete(S, I,1)
    M=np.delete(M, I,0)
    M=np.delete(M, I,1)
    F=np.delete(F, I,0)

    A=S+tau*M
    
    num_iters = 0
    def callback(xk):
        nonlocal num_iters
        num_iters+=1
    
    X, status = cg(A, F, tol=1e-6, callback=callback , maxiter=3000)
    
    te = time.time()
    
    if status == 0:
        success = True
    else:
        success = False
        
    if num_iters == 3000:
        num_iters = 'NC'
        
    newX=reconstruct(X,I)

    error_L2=0
    error_Hcurl=0
    error_energy=0
    for i in range(ntris):
        T=mesh_tris[i]

        p0=mesh_points[T[0]]
        p1=mesh_points[T[1]]
        p2=mesh_points[T[2]]

        Liste=[p0[0],p0[1],p1[0],p1[1],p2[0],p2[1]]
        ndof=r*(r+2)
        Coef=[]
        for j in range(ndof):
            k,sign=local_to_global(nedges, T, tris_edges[i], i, j, r)
            #print("i j k signe",i,j,k,sign)
            Coef.append(sign*newX[k])
        def L2(x,y):
            return (Eval_curl(Liste,r,Coef,x,y)[0]-sol_exct(x,y)[0])**2 +   (Eval_curl(Liste,r,Coef,x,y)[1]-sol_exct(x,y)[1])**2
        def Hcurl(x,y):
            return L2(x,y)+  (Eval_curlcurl(Liste,r,Coef,x,y)-curl_sol_exct(x,y))**2
        def Energy(x,y):
            return tau*L2(x,y)+  (Eval_curlcurl(Liste,r,Coef,x,y)-curl_sol_exct(x,y))**2

        error_L2+=quad(Liste, L2, r+1)
        error_Hcurl+=quad(Liste, Hcurl, r+1)
        error_energy+=quad(Liste,Energy, r+1)

    error_L2=np.sqrt(error_L2)
    error_Hcurl=np.sqrt(error_Hcurl)
    error_energy=np.sqrt(error_energy)

    info={'err_l2_norm': error_L2, 'err_Hcurl_norm': error_Hcurl, 'err_energy':error_energy, 'cond_2': np.linalg.cond(A), 'niter': num_iters, 'success': success, 'CPU_time': te-tb}

    return info

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                             Creates tables
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main_tables(tau,p):
    folder = create_folder(tau  = tau, p=p)
    d = {}
    for k in ks:
        info = main(k      = k,
                    p      = p,
                    tau = tau)

        d[k] = info

    write_table(d, folder, kind ='niter')
    write_table(d, folder, kind ='CPU_time')
    write_table(d, folder, kind ='err_l2_norm')
    write_table(d, folder, kind ='cond_2')
    write_table(d, folder, kind ='err_energy')
    write_table(d, folder, kind ='err_Hcurl_norm')




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               Run tests
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':
    for tau in Taus:
        for p in ps:
            main_tables(tau = tau, p=p)





## TO DO : 
    # make each table concer one couple (tau, p)
    # Add CPU time table
    # K's value :  7 -----> 13
    # K's value for p=1 7 -------> 16



