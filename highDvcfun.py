import numpy as np
import torch
from simplex import euclidean_proj_l1ball as proj1
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
def soft(x, lam):
    return(np.sign(x) * np.clip((np.abs(x) - lam), 0, None))
def thetakj(theta, case, thetaold, partmL, thetakc1, thetakr1, lamr, lamc, p, q, eta):
    if case == 1:
        f= theta- thetaold + eta * partmL + eta * lamr * np.sqrt(q) *  theta /np.sqrt(np.sum(thetakr1**2) + theta**2) + eta * lamc * np.sqrt(p) * theta /np.sqrt(np.sum(thetakc1**2) + theta**2)
        return f
    elif case == 2:
        f= theta - soft(thetaold - eta * partmL, eta * lamr * np.sqrt(q)) + eta * lamc * np.sqrt(p) *  theta /np.sqrt(np.sum(thetakc1**2) + theta**2)
        return f
    elif case == 3:
        f = theta - soft(thetaold - eta * partmL, eta * lamc * np.sqrt(p)) + eta * lamr * np.sqrt(q) * theta /np.sqrt(np.sum(thetakr1**2) + theta**2)
        return f
    else:
        theta = soft(thetaold - eta * partmL, eta * lamc* np.sqrt(p) + eta * lamr * np.sqrt(q))
        return theta
def theta1j(theta, case, thetaold, partmL, thetakc1, lamr, lamc, p, eta):
    if case == 1:
        f= theta - (thetaold - eta * partmL) + eta * lamc * np.sqrt(p) * theta /np.sqrt(np.sum(thetakc1**2) + theta**2)
        return f
    elif case == 2:
        theta = soft(thetaold - eta * partmL, eta * lamc * np.sqrt(p))
        return theta


def estimation(n, p, q, btheta,  maxitr, nsim, eta, lamc, lamr, factor, sf, sd1, sd2):
    mtxtheta = np.zeros((p, q, nsim))
    b0 = np.sqrt(np.sum(btheta**2)) * factor
    s = np.sum(btheta != 0) * factor
    for sitr in range(nsim):
        X = np.random.normal(0, 1, (n, p))
        Z = np.zeros((n, 10))
        temp = np.repeat(range(5), int(n/5))# np.tile(range(q), int(np.ceil(n/q)))[range(n)]
        temp1 = np.random.randint(0, 5, n)
        for j in range(4):
            Z[:, j] =np.random.normal(temp, sd1, n) 
        for j in range(4, 10):
            Z[:, j] =np.random.normal(temp1, sd2, n) 
        #temp = pd.get_dummies(temp, drop_first = True)
        Z = np.hstack((np.ones((n, 1)), Z))
        Z = np.hstack((Z, np.random.normal(0, 1, (n, q - 11))))
        error = np.random.normal(0, 0.5, n)
        #Y = np.diag(np.matmul(np.matmul(X, btheta), Z.transpose())) + error
        design = np.matmul(X[:, :, np.newaxis], Z[:, np.newaxis, :])
        design = design.reshape((n, p * q), order = 'F')
        vtheta = np.reshape(btheta, (p * q, 1), order = 'F')
        Y = np.matmul(design, vtheta)[:, 0] + error
        bthetaini = np.reshape(btheta, (p, q), order = 'F')  + sf *   np.random.uniform(-1, 1, (p, q))
        vthetaini = np.reshape(bthetaini, (p * q), order = 'F')
        #vthetaini = proj1(vthetaini, b0 * np.sqrt(s))
        # L2norm = np.sqrt(np.sum(vthetaini**2))
        # if L2norm  > b0:
        #     vthetaini = vthetaini/L2norm
        bthetaini = np.reshape(vthetaini, (p, q), order = 'F')
        bthetacur =  bthetaini.copy() #btheta.copy() +btheta.copy()#
        for itr in range(maxitr):
            bthetaoo = bthetacur.copy()
            vtheta = np.reshape(bthetacur, (p * q, 1), order = 'F')
            resd = Y - np.matmul(design, vtheta)[:, 0]
            partL = -np.matmul(design.transpose(), resd)/(2 *n)
            partLm = np.reshape(partL, (p, q), order = 'F')
            for i in range(p):
                V = np.zeros(q)
                for j in range(q):
                    if np.sum(np.abs(np.delete(bthetacur[i, :], j))) != 0:
                        V[j] = bthetacur[i, j] - eta * partLm[i, j]
                    else:
                        V[j] = soft(bthetacur[i, j] - eta * partLm[i, j], eta * lamc * np.sqrt(p))
                if(np.sqrt(np.sum(V**2)) <= eta * lamr * np.sqrt(q)):
                    bthetacur[i, :] = 0
            vtheta = np.reshape(bthetacur, (p * q, 1), order = 'F')
            resd = Y - np.matmul(design, vtheta)[:, 0]
            partL = -np.matmul(design.transpose(), resd)/(2 *n)
            partLm = np.reshape(partL, (p, q), order = 'F')
            for j in range(1, q):
                U = np.zeros(p)
                for i in range(p):
                    if np.sum(np.abs(np.delete(bthetacur[:, j], i))) != 0:
                        U[i] = bthetacur[i, j] - eta * partLm[i, j]
                    else:
                        U[i] = soft(bthetacur[i, j] - eta * partLm[i, j], eta * lamr * np.sqrt(q))
                if(np.sqrt(np.sum(U**2)) <= eta * lamc * np.sqrt(p)):
                    bthetacur[:, j] = 0
            index = np.where(bthetacur != 0)
            totIndex = index[0].shape[0]
            for k in range(totIndex):
                i = index[0][k]
                j = index[1][k]
                thetakc1 = np.delete(bthetacur[i, :], j)
                thetakr1 = np.delete(bthetacur[:, j], i)
                delec = np.sum(thetakc1**2)
                deler = np.sum(thetakr1**2)
                partmL = partLm[i, j]
                thetaold = bthetacur[i, j]
                if j== 0 and delec!= 0:
                    case = 1
                    try:
                        res  = optimize.root_scalar(theta1j, args = (case, thetaold, partmL, thetakc1, lamr, lamc, p, eta), bracket =[-10, 10],  method= 'brentq')
                    except:
                        bthetacur[i, j] = bthetacur[i, j]
                    else:
                        bthetacur[i, j] = res.root
                elif j== 0 and delec== 0:
                    case = 2
                    bthetacur[i, j] = theta1j(bthetacur[i, j], case, thetaold, partmL, thetakc1, lamr, lamc, p, eta)
                elif j!=0 and delec != 0 and deler != 0:
                    case = 1
                    try:
                        res = optimize.root_scalar(thetakj, args = (case, thetaold, partmL, thetakc1, thetakr1, lamr, lamc, p, q, eta), bracket =[-10, 10],  method= 'brentq')
                    except:
                        bthetacur[i, j] = bthetacur[i, j]
                    else:
                        bthetacur[i, j] = res.root
                elif j!=0 and deler == 0 and delec !=0:
                    case = 2
                    try:
                        res = optimize.root_scalar(thetakj, args = (case, thetaold, partmL, thetakc1, thetakr1, lamr, lamc, p, q, eta), bracket =[-10, 10],  method= 'brentq')
                    except:
                        bthetacur[i, j] = bthetacur[i, j]
                    else:
                        bthetacur[i, j] = res.root
                elif j!=0 and deler != 0 and delec ==0:
                    case = 3
                    try:
                        res = optimize.root_scalar(thetakj, args = (case, thetaold, partmL, thetakc1, thetakr1, lamr, lamc, p, q, eta), bracket =[-10, 10],  method= 'brentq')
                    except:
                        bthetacur[i, j] = bthetacur[i, j]
                    else:
                        bthetacur[i, j] = res.root
                elif j != 0 and delec == 0 and deler == 0:
                    case = 4
                    bthetacur[i, j] = thetakj(bthetacur[i, j], case, thetaold, partmL, thetakc1, thetakr1, lamr, lamc, p, q, eta)
                vtheta = np.reshape(bthetacur, (p * q, 1), order = 'F')
                resd = Y - np.matmul(design, vtheta)[:, 0]
                partL = -np.matmul(design.transpose(), resd)/(2 *n)
                partLm = np.reshape(partL, (p, q), order = 'F')
            print(np.sum(np.abs(bthetaoo - bthetacur)))
            print(bthetacur)
            vtheta = np.reshape(bthetacur, (p * q), order = 'F')
            vtheta = proj1(vtheta, b0 * np.sqrt(s))
            L2norm = np.sqrt(np.sum(vtheta**2))
            if L2norm  > b0:
                vtheta = vtheta/L2norm
            bthetacur = np.reshape(vtheta, (p, q), order = 'F')
            mtxtheta[:, :, sitr] = bthetacur
            if np.sum(np.abs(bthetaoo - bthetacur)) <=1e-4:
                break
    return mtxtheta, btheta


def estimationcov(n, p, q, btheta,  maxitr, nsim, eta, lamc, lamr, factor, sf, sd1, sd2, sde):
    mtxtheta = np.zeros((p, q, nsim))
    b0 = np.sqrt(np.sum(btheta**2)) * factor
    s = np.sum(btheta != 0)
    for sitr in range(nsim):
        X = np.random.normal(0, sd1, (n, p))
        Z = np.random.normal(0, sd2, (n, p))
        # temp = np.repeat(range(5), int(n/5))# np.tile(range(q), int(np.ceil(n/q)))[range(n)]
        # temp1 = np.random.randint(0, 5, n)
        # for j in range(4):
        #     Z[:, j] =np.random.normal(temp, sd1, n) 
        # for j in range(4, 10):
        #     Z[:, j] =np.random.normal(temp1, sd2, n) 
        # #temp = pd.get_dummies(temp, drop_first = True)
        # Z = np.hstack((np.ones((n, 1)), Z))
        # Z = np.hstack((Z, np.random.normal(0, 1, (n, q - 11))))
        error = np.random.normal(0, sde, n)
        #Y = np.diag(np.matmul(np.matmul(X, btheta), Z.transpose())) + error
        design = np.matmul(X[:, :, np.newaxis], Z[:, np.newaxis, :])
        design = design.reshape((n, p * q), order = 'F')
        vtheta = np.reshape(btheta, (p * q, 1), order = 'F')
        Y = np.matmul(design, vtheta)[:, 0] + error
        bthetaini = np.reshape(btheta, (p, q), order = 'F')  + sf *   np.random.uniform(-1, 1, (p, q)) * np.random.binomial(1, 0.1, (p, q))
        vthetaini = np.reshape(bthetaini, (p * q), order = 'F')
        #vthetaini = proj1(vthetaini, b0 * np.sqrt(s))
        # L2norm = np.sqrt(np.sum(vthetaini**2))
        # if L2norm  > b0:
        #     vthetaini = vthetaini/L2norm
        bthetaini = np.reshape(vthetaini, (p, q), order = 'F')
        bthetacur =  bthetaini.copy() #btheta.copy() +btheta.copy()#
        for itr in range(maxitr):
            bthetaoo = bthetacur.copy()
            vtheta = np.reshape(bthetacur, (p * q, 1), order = 'F')
            resd = Y - np.matmul(design, vtheta)[:, 0]
            partL = -np.matmul(design.transpose(), resd)/(2 *n)
            partLm = np.reshape(partL, (p, q), order = 'F')
            for i in range(p):
                V = np.zeros(q)
                for j in range(q):
                    if np.sum(np.abs(np.delete(bthetacur[i, :], j))) != 0:
                        V[j] = bthetacur[i, j] - eta * partLm[i, j]
                    else:
                        V[j] = soft(bthetacur[i, j] - eta * partLm[i, j], eta * lamc * np.sqrt(p))
                if(np.sqrt(np.sum(V**2)) <= eta * lamr * np.sqrt(q)):
                    bthetacur[i, :] = 0
            vtheta = np.reshape(bthetacur, (p * q, 1), order = 'F')
            resd = Y - np.matmul(design, vtheta)[:, 0]
            partL = -np.matmul(design.transpose(), resd)/(2 *n)
            partLm = np.reshape(partL, (p, q), order = 'F')
            for j in range(1, q):
                U = np.zeros(p)
                for i in range(p):
                    if np.sum(np.abs(np.delete(bthetacur[:, j], i))) != 0:
                        U[i] = bthetacur[i, j] - eta * partLm[i, j]
                    else:
                        U[i] = soft(bthetacur[i, j] - eta * partLm[i, j], eta * lamr * np.sqrt(q))
                if(np.sqrt(np.sum(U**2)) <= eta * lamc * np.sqrt(p)):
                    bthetacur[:, j] = 0
            index = np.where(bthetacur != 0)
            totIndex = index[0].shape[0]
            for k in range(totIndex):
                i = index[0][k]
                j = index[1][k]
                thetakc1 = np.delete(bthetacur[i, :], j)
                thetakr1 = np.delete(bthetacur[:, j], i)
                delec = np.sum(thetakc1**2)
                deler = np.sum(thetakr1**2)
                partmL = partLm[i, j]
                thetaold = bthetacur[i, j]
                if j== 0 and delec!= 0:
                    case = 1
                    try:
                        res  = optimize.root_scalar(theta1j, args = (case, thetaold, partmL, thetakc1, lamr, lamc, p, eta), bracket =[-10, 10],  method= 'brentq')
                    except:
                        bthetacur[i, j] = bthetacur[i, j]
                    else:
                        bthetacur[i, j] = res.root
                elif j== 0 and delec== 0:
                    case = 2
                    bthetacur[i, j] = theta1j(bthetacur[i, j], case, thetaold, partmL, thetakc1, lamr, lamc, p, eta)
                elif j!=0 and delec != 0 and deler != 0:
                    case = 1
                    try:
                        res = optimize.root_scalar(thetakj, args = (case, thetaold, partmL, thetakc1, thetakr1, lamr, lamc, p, q, eta), bracket =[-10, 10],  method= 'brentq')
                    except:
                        bthetacur[i, j] = bthetacur[i, j]
                    else:
                        bthetacur[i, j] = res.root
                elif j!=0 and deler == 0 and delec !=0:
                    case = 2
                    try:
                        res = optimize.root_scalar(thetakj, args = (case, thetaold, partmL, thetakc1, thetakr1, lamr, lamc, p, q, eta), bracket =[-10, 10],  method= 'brentq')
                    except:
                        bthetacur[i, j] = bthetacur[i, j]
                    else:
                        bthetacur[i, j] = res.root
                elif j!=0 and deler != 0 and delec ==0:
                    case = 3
                    try:
                        res = optimize.root_scalar(thetakj, args = (case, thetaold, partmL, thetakc1, thetakr1, lamr, lamc, p, q, eta), bracket =[-10, 10],  method= 'brentq')
                    except:
                        bthetacur[i, j] = bthetacur[i, j]
                    else:
                        bthetacur[i, j] = res.root
                elif j != 0 and delec == 0 and deler == 0:
                    case = 4
                    bthetacur[i, j] = thetakj(bthetacur[i, j], case, thetaold, partmL, thetakc1, thetakr1, lamr, lamc, p, q, eta)
                vtheta = np.reshape(bthetacur, (p * q, 1), order = 'F')
                resd = Y - np.matmul(design, vtheta)[:, 0]
                partL = -np.matmul(design.transpose(), resd)/(2 *n)
                partLm = np.reshape(partL, (p, q), order = 'F')
            print(np.sum(np.abs(bthetaoo - bthetacur)))
            print(bthetacur)
            vtheta = np.reshape(bthetacur, (p * q), order = 'F')
            vtheta = proj1(vtheta, b0 * np.sqrt(s))
            L2norm = np.sqrt(np.sum(vtheta**2))
            if L2norm  > b0:
                vtheta = vtheta/L2norm
            bthetacur = np.reshape(vtheta, (p, q), order = 'F')
            mtxtheta[:, :, sitr] = bthetacur
            if np.sum(np.abs(bthetaoo - bthetacur)) <=1e-4:
                break
    return mtxtheta, btheta

def gentest(n, p, q, nsim, btheta, sd1, sd2):
    mtestY = np.zeros((n, nsim))
    mtestX = np.zeros((n, p, nsim))
    mtestZ = np.zeros((n, q, nsim))
    temp = np.repeat(range(5), int(n/5))# np.tile(range(q), int(np.ceil(n/q)))[range(n)] 
    for sitr in range(nsim):
        X = np.random.normal(0, 1, (n, p))
        Z = np.zeros((n, 10))
        temp1 = np.random.randint(0, 5, n)
        for j in range(4):
            Z[:, j] =np.random.normal(temp, sd1, n) 
        for j in range(4, 10):
            Z[:, j] =np.random.normal(temp1, sd2, n) 
        #temp = pd.get_dummies(temp, drop_first = True)
        Z = np.hstack((np.ones((n, 1)), Z))
        Z = np.hstack((Z, np.random.normal(0, 1, (n, q - 11))))
        error = np.random.normal(0, 0.5, n)
        #Y = np.diag(np.matmul(np.matmul(X, btheta), Z.transpose())) + error
        design = np.matmul(X[:, :, np.newaxis], Z[:, np.newaxis, :])
        design = design.reshape((n, p * q), order = 'F')
        vtheta = np.reshape(btheta, (p * q, 1), order = 'F')
        Y = np.matmul(design, vtheta)[:, 0] + error
        mtestY[:, sitr] = Y
        mtestX[:, :, sitr] = X
        mtestZ[:, :, sitr] = Z
    mtestcl = temp
    return mtestY, mtestX, mtestZ, mtestcl


