#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:38:53 2024

@author: omarch
"""
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
import matplotlib.pyplot as plt

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
d             = {2:4 , 3:8 , 4:16 , 5:44 , 6:101 ,}
Taus          = [10**k for k in range(1,7)]

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                    Creates files which contains results
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def mkdir_p(dir):
    # type: (unicode) -> None
    if os.path.isdir(dir):
        return
    os.makedirs(dir)

def create_folder(tau):

    tau=str(tau)

    folder = 'numerical_results/{tau}'.format(tau=tau)

    mkdir_p(os.path.join(folder, 'txt'))
    mkdir_p(os.path.join(folder, 'tex'))

    return folder

def write_table(d, folder, kind):
    headers = ['#cells/degree p']

    for p in ps:
        headers.append(str(p))

    # add table rows
    rows = []
    for k in ks:
        ncell = d[k]
        row = ['$' + ncell+ '$']
        for p in ps:
            value = d[p, k][kind]
            if isinstance(value, str):
                v = value
            elif isinstance(value, int):
                v = '$'+str(value) +'$'
            else:
                v =  '$'+sprint(value)+'$'
            row.append(v)
        rows.append(row)

    table = tabulate(rows, headers=headers)




    fname = '{label}.txt'.format(label=kind)
    fname = os.path.join('txt', fname)
    fname = os.path.join(folder, fname)

    with open(fname, 'w', encoding="utf-8") as f:
        table = tabulate(rows, headers=headers, tablefmt ='fancy_grid')
        f.write(str(table))

    fname = '{label}.tex'.format(label=kind)
    fname = os.path.join('tex', fname)
    fname = os.path.join(folder, fname)

    with open(fname, 'w') as f:
        table = tabulate(rows, headers=headers, tablefmt='latex_raw')
        f.write(str(table))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                    The main function
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def main(k, p, tau):
    r=p
    print('============tau = {tau}, p = {p}, k = {k}============'
          .format(tau = tau, p=p,k=k))

    mesh_points,mesh_tris,mesh_edges,tris_edges=mesh(k)
    ntris=len(mesh_tris)
    nedges=len(mesh_edges)


    ndof=nbr_globDof(nedges,ntris,r)
    S=np.zeros((ndof,ndof))
    #M=np.zeros((ndof,ndof))

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
        #Mt=mass2d(Liste,r)

        local_ndof=r*(r+2)
        for j in range(local_ndof):
            glob_j,sign_j=local_to_global(nedges,T, tris_edges[i], i ,j,r)
            for k in range(local_ndof):
                glob_k,sign_k=local_to_global(nedges,T, tris_edges[i], i ,k,r)
                S[glob_j][glob_k]+= sign_j * sign_k * St[j][k]
                #M[glob_j][glob_k]+= sign_j * sign_k* Mt[j][k]

    I=IndexToDelete(mesh_edges,mesh_points,r)
    S=np.delete(S, I,0)
    S=np.delete(S, I,1)
    #M=np.delete(M, I,0)
    #M=np.delete(M, I,1)

    #A=S+tau*M

    #info={'ncells': ntris, 'cond_2_stiff': np.linalg.cond(S),'cond_2_Mass': np.linalg.cond(M), 'cond_2_A': np.linalg.cond(A)}
    info={'ncells': ntris,'cond_2_stiff': np.linalg.cond(S)}
    return info

def plot(p, tau):
    lesk=[]
    lesCond=[]
    for k in range(1,12):
        info=main(k, p, tau)
        lesk.append(info['ncells'])
        lesCond.append(info['cond_2_stiff'])
        
    return lesk,lesCond
    

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                             Creates tables
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main_tables(tau):
    folder = create_folder(tau  = tau)
    d = {}
    for p in ps:
        for k in ks:
            info = main(k      = k,
                        p      = p,
                        tau = tau)

            d[p,k] = info

    write_table(d, folder, kind ='err_l2_norm')
    write_table(d, folder, kind ='cond_2')
    write_table(d, folder, kind ='err_energy')
    write_table(d, folder, kind ='err_Hcurl_norm')




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               Run tests
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

'''if __name__ == '__main__':
    for tau in Taus:
        main_tables(tau = tau)'''








