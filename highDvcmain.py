import numpy as np
import torch
from simplex import euclidean_proj_l1ball as proj1
import matplotlib.pyplot as plt
from scipy import optimize
from highDvcfun import soft, thetakj, theta1j, estimation, gentest, estimationcov
from sklearn.cluster import KMeans

mpq = np.array([20, 40, 60, 80])
mn = np.array([200, 500, 1000, 2000, 3000, 4000, 5000])
ntheta1 = []
btheta01 = []
nsim = 50
eta = 0.3
maxitr = 100
sd1 = 0.6
sd2 = 0.3
data = np.load('XYZ.npz')
gX = data['X']
gY = data['Y']
gZ = data['Z']
np.random.seed(0)
temp = np.random.uniform(-3, 3, (80, 80))
for i in range(len(mpq)):
   print(i)
   for j in range(len(mn)):
      print(j)
      p = mpq[i]
      n = mn[j]
      q = mpq[i] 
      btheta = temp.copy()[:p, :q]
      btheta[:, range(5, q)] =  0
      btheta[range(5, p), :] =  0
      lamc = 0.1 *  np.sqrt(np.log(p)/n)
      lamr =  0.1 * np.sqrt(np.log(q)/n)
      tempntheta,  tempbtheta0 = estimationcov(n, p, q, btheta, maxitr, nsim, eta, lamc, lamr, 10, 0, 1, 1, 0.5)
      ntheta1.append(tempntheta)
      btheta01.append(tempbtheta0)
gX = np.hstack((np.ones((gX.shape[0], 1)), gX))
p = np.shape(gX)[1]
q = np.shape(gZ)[1] + 1
n = np.shape(gY)[0]
trainnum = 951
eta = 0.3
nsim = 100
lamc = 0.1 *  np.sqrt(np.log(p)/n) #0.5 *  np.sqrt(np.log(p)/n)
lamr =0.1  *  np.sqrt(np.log(p)/n)# 0.5 * np.sqrt(np.log(q)/n)
lambda_seq = np.arange(0, 0.003, 0.001)
temp = np.random.uniform(-3, 3, (p, q))
btheta = temp.copy()
btheta[:, :] =  0
btheta[np.array([0, 10, 20, 80]), 1] =  np.random.uniform(-3, 3, 4)
btheta[np.array([0, 10, 20, 80]), 0] =  np.random.uniform(-0.2, 0.2, 4)
tempntheta,  tempbtheta0, trainix = estimation(n, p, q,1000, nsim, eta, lamc, lamr, 4, 0.1, sd1, sd2, gX, gZ, gY, trainnum, False,lambda_seq, btheta, startIx = 1,  ifsimu = False)

tempntheta,  tempbtheta0, trainix = estimation(n, p, q,1000, nsim, eta, lamc, lamr, 4, 0.1, sd1, sd2, gX, gZ, gY, trainnum, False,lambda_seq, btheta, startIx = 1,  ifsimu = True)

allntheta,  allntheta0, trainix = estimation(n, p, q,1000, nsim, eta, lamc, lamr, 4, 0.1, sd1, sd2, gX, gZ, gY, trainnum, False,lambda_seq, btheta, startIx = 0,  ifsimu = True)

no1stntheta,  no1stntheta0, trainix = estimation(n, p, q,1000, nsim, eta, lamc, lamr, 4, 0.1, sd1, sd2, gX, gZ, gY, trainnum, False,lambda_seq, btheta, startIx = 1,  ifsimu = True)

lesstntheta,  lessstntheta0, trainix = estimation(n, p, q,1000, nsim, eta, lamc, lamr, 10000, 0.1, sd1, sd2, gX, gZ, gY, trainnum, False,lambda_seq, btheta, startIx = 1,  ifsimu = True)

np.savez('simsden', ntheta = tempntheta, btheta0 = tempbtheta0,  allntheta = allntheta, btheta = btheta, no1stntheta = tempntheta)

      ntheta1.append(tempntheta)
      btheta01.append(tempbtheta0)

mtestY, mtestX, mtestZ  =  gentest(1901 - trainnum, p, q, nsim, tempntheta, sd1, sd2, gX, gZ, gY, trainix)
n = mtestY.shape[0]
ncl = 5
kmest = np.zeros((n, nsim))
estcl = np.zeros((ncl, p, nsim))
swestcl = np.zeros((ncl, p, nsim))
kmtrue = np.zeros((n, nsim))
truecl = np.zeros((ncl, p, nsim))
mestix = np.zeros((p, nsim))
mtrueix = np.zeros((p, nsim))
swkmest = np.zeros((n, nsim))
for itr in range(nsim):
   temp = np.matmul(mtestZ[:, 1:, itr], tempntheta[:, 1:, itr].transpose())
   inclix = np.sum(np.abs(temp), 0) >= 0
   ix = np.where(inclix)[0]
   temp = KMeans(n_clusters=5, random_state=0).fit(temp[:, ix])
   #kmtrue[:, itr] = temp.labels_
   #truecl[:, ix, itr] = temp.cluster_centers_
   idx = np.argsort(temp.cluster_centers_.sum(axis=1))
   lut = np.zeros_like(idx)
   lut[idx] = np.arange(ncl)
   kmtrue[:, itr] = lut[temp.labels_]
   truecl[:, ix, itr] = temp.cluster_centers_[idx, :]
   mtrueix[:, itr]= inclix
   temp = np.matmul(mtestZ[:, 1:, itr], tempntheta[:, 1:, itr].transpose())
   inclix = np.sum(np.abs(temp), 0) >= 0
   ix = np.where(inclix)[0]
   temp = KMeans(n_clusters=5, random_state=0).fit(temp[:, ix])
   idx = np.argsort(temp.cluster_centers_.sum(axis=1))
   lut = np.zeros_like(idx)
   lut[idx] = np.arange(ncl)
   kmest[:, itr] = lut[temp.labels_]
   estcl[:, ix, itr] = temp.cluster_centers_[idx, :]
   mestix[:, itr]= inclix

np.savez('cluster100_sd_0.5', ntheta = tempntheta, btheta0 = tempbtheta0, kmtrue = kmtrue, kmest= kmest, estcl = estcl, truec = truecl, mtestZ = mtestZ,mn =  mn, mpq = mpq, sd1 = sd1, sd2 = sd2)

np.savez('realdataresX_h', ntheta = tempntheta, btheta0 = tempbtheta0,  trainix = trainix)
np.savez('crossresspline', ntheta = tempntheta, btheta0 = tempbtheta0,  trainix = trainix)
np.savez('testingX_h', mtestY= mtestY, mtestX= mtestX, mtestZ = mtestZ)
clusterres = np.load('cluster100_sd_0.5.npz')

tempbtheta0 =clusterres['btheta0']
sd1 = clusterres['sd1']
mtestY, mtestX, mtestZ, mtestcl =  gentest(n, p, q, nsim, tempbtheta0, sd1, sd2)



for itr in range(nsim):
   for i in range(ncl):
      dist = np.sum((truecl[:, :, itr] - estcl[i, :,   itr][np.newaxis, :] )**2, 1)
      ix = np.where(dist == np.min(dist))[0]
      swkmest[kmest[:, itr] == i, itr] = ix
      swestcl[i, :, itr] = estcl[ix, :, itr]
np.sum(abs(kmtrue - kmest) >0, 0)/n
   
   
np.savez('p_20_100_dis', ntheta = ntheta1, btheta = btheta01, mn = mn, mpq = mpq)
k = 0 
fnorm = np.zeros((len(mpq), len(mn)))
for i in range(len(mpq)):
   print(i)
   for j in range(len(mn)):
      print(j)
      estk = ntheta1[k]
      truek = btheta01[k]
      fnorm[i, j] = np.median(np.sqrt(np.sum((estk - truek[:, :, np.newaxis]) **2)))
      k = k + 1
k = 0 
fnorm100 = np.zeros((1, len(mn)))
for i in range(1):
   print(i)
   for j in range(len(mn)):
      print(j)
      estk = ntheta[k]
      truek = btheta0[k]
      fnorm100[i, j] = np.median(np.sqrt(np.sum((estk - truek[:, :, np.newaxis]) **2)))
      k = k + 1
fnorm_20_100= np.vstack((fnorm, fnorm100))
np.save('fnorm_20_120', fnorm)
ntheta[:, :, :, nitr] = mtxtheta
mserror = np.zeros(7)
for nitr in range(7):
   mserror[nitr] = np.sqrt(np.sum((ntheta[:, :, :, nitr] - btheta[:, :, np.newaxis])**2))
o = np.argsort(btheta.reshape(p * q, 1)[:, 0])
nitr = 4
plt.plot(range(p * q), btheta.reshape((p * q),  order = 'F')[o], color = 'b')
plt.plot(range(p * q), np.percentile(ntheta[:, :, :, nitr], q= 0.975, axis =  2).reshape((p * q), order = 'F')[o], color = 'g')

plt.plot(range(p * q), np.percentile(ntheta[:, :, :, nitr], q= 0.025, axis =  2).reshape((p * q), order = 'F')[o], color = 'g')
v
plt.plot(range(p * q), np.percentile(ntheta[:, :, :, nitr], q= 0.05, axis =  2).reshape((p * q), order = 'F')[o], color = 'r')
plt.savefig('estimator100')
