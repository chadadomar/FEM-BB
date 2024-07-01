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


ps            = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
ks            = (2, 3, 4, 5, 6)

#ps            = (1,2,3,4,5,6,7,8)
#ks            = (7,8) #9,10,11,12,13
ncells        = {2:"4", 3:"8", 4:"16", 5:"44", 6:"101", 7:"215", 8:"401", 9:"800", 10:"1586", 11:"3199", 12:"6354", 13:"12770", 14:"25497", 15:"50917", 16:"101741", 17:"203504", 18:"406760"}
Taus          = [10**k for k in range(-4,5)]

# Right hand side and solution
def u2(x,y):
    return np.sin(np.pi*y) * np.sin(np.pi*x)

def gradu2(x,y):
    return np.array( [  np.pi*np.sin(np.pi*y)*np.cos(np.pi*x), np.pi*np.sin(np.pi*x)*np.cos(np.pi*y) ] )

def scd2(x,y):
    return np.pi**2*u2(x,y)

def scd1(x,y):
    return 2*(x*(1-x)+y*(1-y))

def u1(x,y):
    return x*(1-x)*y*(1-y)

def gradu1(x,y):
    return np.array([-2*x*y*(1-y),-2*x*y*(1-x)])

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

#check if a point is on the boundary
def Onboundary(P):
    x=P[0]
    y=P[1]
    if x*y*(1-x)*(1-y)==0:
        return True
    else:
        return False

#check if an edge is on the boundary of the unit square [0,1]^2
def EdgeOnboundary(v,u):
    flag=False
    if v[0]==u[0]:
        if v[0]==0 or v[0]==1:
            flag=True
    elif v[1]==u[1]:
        if v[1]==0 or v[1]==1:
            flag=True
    return flag

# Collect indices of globale functions non vanishing on boundary of [0,1]^2
def IndexToDelete_H1(mesh_edges,mesh_points,p):
    # retrun sorted liste of indices of global functions non vanishing on boundary of [0,1]^2
    I=[]
    nedges=len(mesh_edges)
    nvertices=  len(mesh_points)
    for i in range(nvertices):
        if Onboundary(mesh_points[i]):
            I.append(i)
    for i in range(nedges):
        E=mesh_edges[i]
        p1=mesh_points[E[0]]
        p2=mesh_points[E[1]]
        if EdgeOnboundary(p1,p2):
            for j in range(p-1):
                ind=nvertices + i*(p-1)+j
                I.append(ind)
    return I


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


def main(k, p):
    print('============ p = {p}, k = {k}============'
          .format(tau = tau, p=p,k=k))
    
    tb = time.time()

    mesh_points,mesh_tris,mesh_edges,tris_edges=mesh(k)
    nvertices=len(mesh_points)          # number of domaine points  
    nedges= len(mesh_edges)
    ntris = len(mesh_tris)
    ndof=nbr_glob_Dof(nvertices,nedges,ntris,p) # global dof for H1



    M=np.zeros((ndof,ndof))             # Golbal mass matrix
    S=np.zeros((ndof,ndof))             # Golbal stiffness matrix
    B=np.zeros(ndof)                    # Global load vector    
    w=(p+2)*(p+1)//2


    for ti in range(ntris):
        t=mesh_tris[ti]
        # vertex of the triangle/elem t
        v0=mesh_points[t[0]]
        v1=mesh_points[t[1]]
        v2=mesh_points[t[2]]
        ##liste of vertices's coordinates
        Trig=[v0[0],v0[1],v1[0],v1[1], v2[0],v2[1]] 
        Se=cst_StiffMat_2D(Trig,np.eye(2), p)       #local stifness matrix
        Be=Moment2D(Trig, scd1 , p)                  #local load vector
        Me=MassMat2D(Trig,lambda x,y:1,p)           #local stifness matrix
        Fe=Moment2D(Trig, u1 , p)                  #local load vector

        for i in range(w):
            I=local_to_global_H1(nvertices, nedges, t, tris_edges[ti], ti, i, p)
            B[I]+=Be[i]
            F[I]+=Fe[i]
            for j in range(w):
                J=local_to_global_H1(nvertices, nedges, t, tris_edges[ti], ti, j, p)
                S[I][J]+=Se[i][j]
                M[I][J]+=Me[i][j]
                    
    
    I=IndexToDelete_H1(mesh_edges,mesh_points,p)
    S=np.delete(S, I,0)
    S=np.delete(S, I,1)
    B=np.delete(B, I,0)
    A=S
    U=np.linalg.solve(M,F)
    
    num_iters = 0
    def callback(xk):
        nonlocal num_iters
        num_iters+=1
    
    X, status = cg(A, B, tol=1e-6, callback=callback , maxiter=3000)
    
    te = time.time()
    
    if status == 0:
        success = True
    else:
        success = False
        
    if num_iters == 3000:
        num_iters = 'NC'
        
    C=reconstruct(X,I)
    
    error_L2_projection=0
    error_L2=0
    error_H1=0
    for ti in range(ntris):
        t=mesh_tris[ti]
        # vertex of the triangle/elem t
        v0=mesh_points[t[0]]
        v1=mesh_points[t[1]]
        v2=mesh_points[t[2]]
        ##liste of vertices's coordinates
        Trig=[v0[0],v0[1],v1[0],v1[1], v2[0],v2[1]] 
        
        def L2_projection(x,y):
            lam=BarCord2d(Trig,x,y)
            BB=[]
            for j in range(w):
                J=local_to_global_H1(nvertices, nedges, t,  tris_edges[ti], ti, j, p)
                BB.append(U[J])
            return (u1(x,y)-deCasteljau2D(lam,BB,p))**2
        
        def L2(x,y):
            lam=BarCord2d(Trig,x,y)
            BB=[]
            for j in range(w):
                J=local_to_globalH1(nvertices, nedges, t,  tris_edges[ti], ti, j, p)
                BB.append(C[J])
            return (u1(x,y)-deCasteljau2D(lam,BB,p))**2
        
        def H1(x,y):
            return L2(x,y) +  (Eval_grad(Trig,p,BB,x,y)-gradu1(x,y))[0]**2 + (Eval_grad(Trig,p,BB,x,y)-gradu1(x,y))[1]**2


        error_L2+=quad(Trig, L2, p)
        error_L2_projection+=quad(Trig,L2_projection,p)
        error_H1+=quad(Trig, H1, p)

    error_L2=np.sqrt(error_L2)
    error_L2_projection=np.sqrt(error_L2_projection)
    error_H1=np.sqrt(error_H1)

    info={'err_L2_projection':error_L2_projection, 'err_l2_norm': error_L2, 'err_H1_norm': error_H1, 'cond_2': np.linalg.cond(A), 'niter': num_iters, 'success': success, 'CPU_time': te-tb}

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

'''if __name__ == '__main__':
    for tau in Taus:
        for p in ps:
            main_tables(tau = tau, p=p)'''





## TO DO : 
    # make each table concer one couple (tau, p)
    # Add CPU time table
    # K's value :  7 -----> 13
    # K's value for p=1 7 -------> 16



