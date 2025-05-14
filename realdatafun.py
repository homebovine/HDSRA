import numpy as np
import torch
from simplex import euclidean_proj_l1ball as proj1
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn import linear_model
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
def theta1j(theta, case, thetaold, partmL, thetakc1, lamr, lamc, q, eta):
    if case == 1:
        f= theta - (thetaold - eta * partmL) + eta * lamr * np.sqrt(q) * theta /np.sqrt(np.sum(thetakc1**2) + theta**2)
        return f
    elif case == 2:
        theta = soft(thetaold - eta * partmL, eta * lamr * np.sqrt(q))
        return theta


def estimation(n, p, q, maxitr, nsim, eta, lamc, lamr, factor, sf, sd1, sd2, gX, gZ, gY, trainnum, replace, lambda_seq, btheta, startIx, ifsimu = False):
    mtxtheta = np.zeros((p, q, nsim))
    mtxthetaini = np.zeros((p, q, nsim))
    
    b0 = np.sqrt(np.sum(btheta**2)) * factor
    s = np.sum(btheta != 0) * factor
    nbig = np.shape(gX)[0]
    ixcontain = np.zeros((nsim, trainnum))
    for sitr in range(nsim):
        if ifsimu == True:
            X = gX#np.random.normal(0, 1, (n, p))
            Z = gZ#np.zeros((n, 10))
        else:
            if sitr == 0:
                trainix = np.arange(0, trainnum)
            else:
                trainix = np.random.choice(nbig, trainnum, replace=replace)
            X = gX[trainix, :]
            Z= gZ[trainix, :]
            ixcontain[sitr, :] = trainix
        n = np.shape(X)[0]
        Z = np.hstack((np.ones((n, 1)), Z))
        error = np.random.normal(0, 0.5, n)
        #Y = np.diag(np.matmul(np.matmul(X, btheta), Z.transpose())) + error
        design = np.matmul(X[:, :, np.newaxis], Z[:, np.newaxis, :])
        design = design.reshape((n, p * q), order = 'F')
        vtheta = np.reshape(btheta, (p * q, 1), order = 'F')
        if ifsimu == True:
            Y = np.matmul(design, vtheta)[:, 0] + error
        #    bthetaini = np.reshape(btheta, (p, q), order = 'F')  + sf *   np.random.uniform(-1, 1, (p, q))
         #   vthetaini = np.reshape(bthetaini, (p * q), order = 'F')
        else: 
            Y = gY[trainix, 0]
        reg = linear_model.LassoCV(precompute=True, cv=5, verbose=1, n_jobs=5, alphas = lambda_seq)
        reg.fit(design, Y)
        alpha = reg.alpha_
        #print(alpha)
        clf = linear_model.Lasso(alpha=alpha, fit_intercept=False)
        clf.fit(design, Y)
        vthetaini = clf.coef_
        b0 = np.sqrt(np.sum(vthetaini**2)) * factor
        s = np.sum(vthetaini != 0) * factor
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
            for j in range(startIx, q):
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
                        res  = optimize.root_scalar(theta1j, args = (case, thetaold, partmL, thetakc1, lamr, lamc, q, eta), bracket =[-10, 10],  method= 'brentq')
                    except:
                        bthetacur[i, j] = bthetacur[i, j]
                    else:
                        bthetacur[i, j] = res.root
                elif j== 0 and delec== 0:
                    case = 2
                    bthetacur[i, j] = theta1j(bthetacur[i, j], case, thetaold, partmL, thetakc1, lamr, lamc, q, eta)
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
            print(bthetacur[0, :])
            vtheta = np.reshape(bthetacur, (p * q), order = 'F')
            vtheta = proj1(vtheta, b0 * np.sqrt(s))
            L2norm = np.sqrt(np.sum(vtheta**2))
            if L2norm  > b0:
                vtheta = vtheta/L2norm
            bthetacur = np.reshape(vtheta, (p, q), order = 'F')
            mtxtheta[:, :, sitr] = bthetacur
            mtxthetaini[:, :, sitr] = bthetaini
            if np.sum(np.abs(bthetaoo - bthetacur)) <=1e-4:
                break
    return mtxtheta, mtxthetaini, ixcontain

def gentest(n, p, q, nsim, btheta, sd1, sd2, gX, gZ, gY, ixcontain, ifsimu = False):
    mtestY = np.zeros((n, nsim))
    mtestX = np.zeros((n, p, nsim))
    mtestZ = np.zeros((n, q, nsim))
    for sitr in range(nsim):
        if ifsimu == True:
            X = gX#np.random.normal(0, 1, (n, p))
            Z = gZ#np.zeros((n, 10))
        else:
            ix = np.delete(np.arange(gX.shape[0]), ixcontain[sitr, :])
            X = gX[ix, :]
            Z= gZ[ix, :]
        n = np.shape(X)[0]
        Z = np.hstack((np.ones((n, 1)), Z))
        if ifsimu == True:
            Y = np.matmul(design, vtheta)[:, 0] + error
        else: 
            Y = gY[ix, 0]
        mtestY[:, sitr] = Y
        mtestX[:, :, sitr] = X
        mtestZ[:, :, sitr] = Z
    return mtestY, mtestX, mtestZ


