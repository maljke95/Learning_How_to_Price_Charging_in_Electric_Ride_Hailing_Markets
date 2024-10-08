
import numpy as np
from scipy.linalg import block_diag
import cvxpy as cp
from quadprog import solve_qp
import os
from datetime import datetime 
import cvxopt
from cvxopt import matrix, solvers
    
import matplotlib.pyplot as plt
from matplotlib import cm
import tikzplotlib

from scipy.optimize import LinearConstraint
from scipy.optimize import milp


if __name__ == '__main__':
    
    load_folder = os.getcwd() + '/Parameters_Load'
    
    N1 = np.load(load_folder + '/N1.npy')
    N2 = np.load(load_folder + '/N2.npy')
    N3 = np.load(load_folder + '/N3.npy')    
    
    Ni_list = [N1, N2, N3]
    
    G1 = np.load(load_folder + '/G1.npy')
    A1 = np.load(load_folder + '/A1.npy')
    b1 = np.array(np.load(load_folder + '/b1.npy'))
    h1 = np.squeeze(np.load(load_folder + '/h1.npy'))
    K_l1 = np.load(load_folder + '/K_l1.npy')
    k_r1 = np.load(load_folder + '/k_r1.npy')
    
    G2 = np.load(load_folder + '/G2.npy')
    A2 = np.load(load_folder + '/A2.npy')
    b2 = np.array(np.load(load_folder + '/b2.npy'))
    h2 = np.squeeze(np.load(load_folder + '/h2.npy'))
    K_l2 = np.load(load_folder + '/K_l2.npy')
    k_r2 = np.load(load_folder + '/k_r2.npy')
    
    G3 = np.load(load_folder + '/G3.npy')
    A3 = np.load(load_folder + '/A3.npy')
    b3 = np.array(np.load(load_folder + '/b3.npy'))
    h3 = np.squeeze(np.load(load_folder + '/h3.npy'))   
    K_l3 = np.load(load_folder + '/K_l3.npy')
    k_r3 = np.load(load_folder + '/k_r3.npy')
    
    r1 = np.squeeze(np.load(load_folder + '/r1.npy'))
    r2 = np.squeeze(np.load(load_folder + '/r2.npy'))
    r3 = np.squeeze(np.load(load_folder + '/r3.npy'))
    
    S1 = np.load(load_folder + '/S1.npy')
    S2 = np.load(load_folder + '/S2.npy')
    S3 = np.load(load_folder + '/S3.npy')

    G = np.array(block_diag(G1, G2, G3))
    h = np.concatenate((np.concatenate((h1, h2)), h3))
    
    K_l = np.array(block_diag(K_l1, K_l2, K_l3))
    k_r = np.concatenate((np.concatenate((k_r1, k_r2)), k_r3))

    
    P1  = np.load(load_folder + '/P1.npy')/N1**2
    P2  = np.load(load_folder + '/P2.npy')/N2**2
    P3  = np.load(load_folder + '/P3.npy')/N3**2                               # They are all the same
    P_list   = [P1, P2, P3]
    
    Q1  = np.load(load_folder + '/Q1.npy')/N1
    Q2  = np.load(load_folder + '/Q2.npy')/N2
    Q3  = np.load(load_folder + '/Q3.npy')/N3
    Q   = [Q1, Q2, Q3]
    
    N_des = np.load(load_folder + '/N_des.npy')
    
    #----- CDC list of vars -----
    A = np.array(block_diag(A1, A2, A3))
    b = np.concatenate((np.concatenate((b1*N1, b2*N1)), b3*N1))
    
    P = P1
    C = Q1
    
    r1 = np.squeeze(np.load(load_folder + '/r1.npy'))/N1
    r2 = np.squeeze(np.load(load_folder + '/r2.npy'))/N2
    r3 = np.squeeze(np.load(load_folder + '/r3.npy'))/N3
    
    S1 = np.load(load_folder + '/S1.npy')/N1
    S2 = np.load(load_folder + '/S2.npy')/N2
    S3 = np.load(load_folder + '/S3.npy')/N3
    
    c_ovr = 0.4
    N_ovr = 194
    N_und = 157
    N_tot = 194+157+182
    
    P_1 = np.linalg.inv(P)
    alpha = np.trace(P_1)**(-1)
    
    onesv = np.ones((4,1))
    psi = np.eye(4)-alpha* onesv @ onesv.T @ P_1
    
    r1ov = psi @ r1
    r2ov = psi @ r2
    r3ov = psi @ r3
    
    rov = np.array(list(r1ov)+list(r2ov)+list(r3ov))
    
    rmax = np.max(rov)
    rmin = np.min(rov)
    
    zmax = (c_ovr-alpha/2.0)*N_tot + (c_ovr+alpha/2.0)*N_ovr
    zmin = -alpha/2.0*(N_und-N_tot)
    
    g_little = alpha*N_und - rmax -zmax
    g_big = alpha*N_ovr - rmin -zmin
    
    psis1 = psi @ S1
    psis2 = psi @ S2
    psis3 = psi @ S3
    
    Gpi = np.concatenate((psis1, psis2), axis=0)
    Gpi = np.concatenate((Gpi,psis3), axis=0)
    
    G_constr = np.concatenate((Gpi, -Gpi), axis=0)
    h_constr = np.concatenate((g_big*np.array(12*[1.0]), g_little*np.array(12*[-1.0])), axis=0)
    
    #----- Constraint -----
    
    # pi = cp.Variable(4)
    # vec = np.array([0.0, 0.0, 0.0, 1.0]).reshape(-1,1)
    
    # prob = cp.Problem(cp.Minimize( vec.T @ pi),
    #                       [Gpi/5 @ pi <= g_big/5*np.array(12*[1.0])])

    # prob.solve()
    
    # print(pi.value)
    
    #----- Theorem 2 -----
    onesv3 = np.ones((3,1))
    
    L = G1.shape[0] + G1.shape[0] + G1.shape[0]
    P1_b = np.kron(np.eye(3),C) + np.kron(onesv3 @ onesv3.T, C)
    P2_b = np.array(block_diag(A1.T, A2.T, A3.T))
    P3_b = np.array(block_diag(G1.T, G2.T, G3.T))
    r_b  = np.concatenate((np.concatenate((r1,r2), axis=0), r3), axis=0)
    A_b  = A
    b_b  = np.array(Ni_list)
    G_b  = np.array(block_diag(G1, G2, G3))
    h_b  = np.concatenate((np.concatenate((h1*N1,h2*N2), axis=0), h3*N3), axis=0) 
    S_b  = np.concatenate((np.concatenate((S1,S2), axis=0), S3), axis=0)
    
    # c = -np.array([0, 1])
    # A = np.array([[-1, 1], [3, 2], [2, 3]])
    # b_u = np.array([1, 12, 12])
    # b_l = np.full_like(b_u, -np.inf)
    # constraints = LinearConstraint(A, b_l, b_u)
    # integrality = np.ones_like(c)
    # res = milp(c=c, constraints=constraints, integrality=integrality)
    # print(res.x)
    
    Big_constr_L1 = np.hstack([P1_b, P2_b, P3_b, -S_b, np.zeros((12,L))])
    Big_constr_r1 = r_b
    
    Big_constr_L2 = np.hstack([A_b, np.zeros((3,3)), np.zeros((3,54)), np.zeros((3,4)), np.zeros((3,54))])
    Big_constr_r2 = b_b
    
    Lambda = np.kron(np.ones((1,3)), np.eye(4))
    N_des = np.load(load_folder + '/N_des.npy')
    
    Big_constr_L3 = np.hstack([Lambda, np.zeros((4,3)), np.zeros((4,54)), np.zeros((4,4)), np.zeros((4,54))])
    Big_constr_r3 = N_des
    
    Eq_L = np.vstack([Big_constr_L1, Big_constr_L2, Big_constr_L3])
    Eq_r = np.vstack([Big_constr_r1.reshape(-1,1), Big_constr_r2.reshape(-1,1), Big_constr_r3.reshape(-1,1)]).squeeze()
    
    #----- Ineq constr ---
    
    beta = 100.0
    
    Big_in_L1 = np.hstack([np.zeros((54,12)), np.zeros((54,3)), np.eye(54), np.zeros((54,4)), -beta*np.eye(54)])
    Big_in_r1 = np.zeros((54,1))
    
    Big_in_L2 = np.hstack([np.zeros((54,12)), np.zeros((54,3)), -np.eye(54), np.zeros((54,4)), np.zeros((54,54))])
    Big_in_r2 = np.zeros((54,1))
    
    Big_in_L3 = np.hstack([-G_b, np.zeros((54,3)), np.zeros((54,54)), np.zeros((54,4)), beta*np.eye(54)])
    Big_in_r3 = -h_b.reshape(-1,1) +beta*np.ones((54,1))
    
    Big_in_L4 = np.hstack([G_b, np.zeros((54,3)), np.zeros((54,54)), np.zeros((54,4)), 0.0*np.eye(54)])
    Big_in_r4 = h_b.reshape(-1,1) 
    
    In_L = np.vstack([Big_in_L1, Big_in_L2, Big_in_L3, Big_in_L4])
    In_r = np.vstack([Big_in_r1, Big_in_r2, Big_in_r3, Big_in_r4]).reshape(-1,1)
    
    integrality = np.zeros(127)
    for cnt in range(73,127):
        integrality[cnt] = 1
        
    b_l     = np.vstack([Big_constr_r1.reshape(-1,1), Big_constr_r2.reshape(-1,1), Big_constr_r3.reshape(-1,1), np.full_like(In_r, -np.inf)])
    b_u     = np.vstack([Big_constr_r1.reshape(-1,1), Big_constr_r2.reshape(-1,1), Big_constr_r3.reshape(-1,1), In_r                       ])
    A_scipy = np.vstack([Big_constr_L1, Big_constr_L2, Big_constr_L3, In_L                       ])
    
    constraints = LinearConstraint(A_scipy, b_l.squeeze(), b_u.squeeze())
    c = np.zeros(127)
    
    res = milp(c=c, constraints=constraints, integrality=integrality)