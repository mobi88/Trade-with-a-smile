import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima_process import ArmaProcess
import statsmodels.formula.api as smf

import statsmodels.api as sm
import scipy.stats as scs

import scipy as sp
import functools
import operator
import itertools
from math import pi
##
dsfm= pd.read_csv("C:/Users/mobi/iCloudDrive/3L68KQB4HG~com~readdle~CommonDocuments/All Materials/5-Master Thesis/untitled/dsfm.csv")
dsfm= dsfm.iloc[4:]
dsfm.columns= ["data_date","iv","LM","TTM"]

##
# date = np.unique(dsfm["data_date"])
L=3
L0   = L+1
numDataPoints= 25
h=[0.5,0.5]
tol = 1e-5
maxIt = 301
# I= len(date)

##
### amended ######
date = np.unique(dsfm["data_date"])
I    = len(date)
L0   = L+1
Y    = []
x2   = [] #### Time to Maturity
x1   = [] #### Moneyness
J  = []
for i in range(0, I):
    z2 = np.array(np.unique(dsfm.loc[dsfm["data_date"] == date[i]]["TTM"]))
    z1 = np.array(np.unique(dsfm.loc[dsfm["data_date"] == date[i]]["LM"]))
    y = np.array((dsfm.loc[dsfm["data_date"] == date[i]]["iv"]))
    j = len(z1) * len(z2)
    x2.append(z2)
    x1.append(z1)
    J.append(j)
    Y.append((np.asmatrix(y).reshape(len(z2),len(z1))))

##
for i in range(0, len(x1)):### lapply(x1, function(x) x / max(x))
    x1[i] = x1[i] / x1[i].max()
for i in range(0, len(x2)):
    x2[i] = x2[i] / x2[i].max()
##
### amended #####
alphas = np.array([0.4])
betas = np.array([0.0])
ar = np.r_[1, alphas]
ma = np.r_[1, betas]
ZHat=np.asmatrix(np.zeros((I,L)))
for i in range(0, L):
    ZHat[:,i] = np.asmatrix(smt.arma_generate_sample(ar=ar, ma=ma, nsample=I)).T
ZHat= np.concatenate( ( (np.ones((I,1))) , ZHat), axis=1)
##
# numDataPoints=25
##
minx1=np.min([y for x in x1 for y in x])
minx2=np.min([y for x in x2 for y in x])
maxx1=np.max([y for x in x1 for y in x]) # trivial, the max is always 1 since we standardized all of them
maxx2=np.max([y for x in x2 for y in x])
delta=((maxx1 - minx1) * (maxx2 - minx2)) / numDataPoints**2
u1= np.linspace(minx1, maxx1, num=numDataPoints)
u1Points= np.linspace(minx1, maxx1, num=numDataPoints)
u2 = np.linspace(minx2, maxx2, num=numDataPoints)
u2Points= np.linspace(minx2, maxx2, num=numDataPoints)
u1Points=np.tile(u1Points, len(u2Points))
u1Points=np.sort(u1Points)
u2Points=np.tile(u2Points, len(u2Points))
u = pd.DataFrame({'u1Points': u1Points, 'u2Points': u2Points})
U= len(u1)*len(u2)
##
# Loop parameters
# procede to next block to avoid matrix class (deprecated from numpy)
ZHatOld= np.matrix(np.zeros((I,L0)))
mHatOld= np.matrix(np.zeros((U,L0)))
it= 0
stopCriterion= 1
plotStopCriterion=[]
##
ZHatOld= np.array(np.zeros((I,L0)))
mHatOld= np.array(np.zeros((U,L0)))
it= 0
stopCriterion= 1
plotStopCriterion= []
h=[0.3,0.4]
##
def quartickernel1d(x):  ### why calculate the kernel only for entries <= 1
    quartic1D = np.where(abs(x)<=1, np.array(((15/16)*(1-x**2)**2)), 0  )
    return quartic1D
##
def kerneldensity2d(y, I, J, x1, x2, u, U, h):
    pTHat = np.zeros((I,U))
    qTHat = jQTHat = jPTHat = pTHat
##
## amended version
## Kernel function
pTHat = []
qTHat = []
jQTHat = []
jPTHat = []
summ = lambda x: x.sum()
# jPTHat=np.array(jPTHat)
for t in range(0, I):
    pTjHatList = []
    qTjHatList = []
    for n in range(0, U):
        o= np.dot(np.atleast_2d(((1/h[0])*quartickernel1d((u.u1Points.iloc[n]-x1[t])/h[0]))).T , np.atleast_2d((1/h[1]) * (quartickernel1d((u.u2Points.iloc[n]-x2[t])/h[1]))))
        pTjHatList.append(np.asmatrix(o))  ## pre --Kernel density estimate--
                                            ## T because it is still not summed, j because it is still not devided by j
        l = np.multiply(o,Y[t].T)
        qTjHatList.append(np.asmatrix(l))  ## Pre --Weighted sum of the dependent variable.

    new = list(map((summ), pTjHatList))
    jPTHat.append(new)  ## pre kernel density (need to be devided by j(i) )
    new = list(map(lambda x:x/J[t],(list(map(summ,pTjHatList)))))
    pTHat.append(new)  ## should sum every matrix in u*u dimension and devide it by Jt
    new1 = list(map(summ, qTjHatList))
    jQTHat.append(new1)
    new1 = list(map(lambda x:x/J[t],(list(map(summ,qTjHatList)))))
    qTHat.append(new1)
##
#  Matrix B
##   amended  ####
Bu = np.zeros(( L0, L0))
BList = np.tile(Bu, (U, 1,1))
for n in range(0,U):
#     ZHat1=np.multiply(ZHat,(np.atleast_2d(jPTHat1[:,1]).T))
    BList[n]= np.dot(ZHat.T,  np.multiply(ZHat,np.atleast_2d([j[n] for j in jPTHat]).T)  )
##
# Vector Q
### amended ###
# Q = np.zeros((1,L0))
QList = []
for n in range(0,U):
    QList.append([])
    QList[n]= np.dot(ZHat.T,np.atleast_2d([j[n] for j in jQTHat]).T)
##
### amended ####
# Vector mHat for each grid points of length L

mHat = np.array(np.zeros((U, L0)))
for i in range(0, U):
    k = np.linalg.inv(BList[i])
    m = np.dot(k, QList[i])
    mHat[i, :] = m.T
### since matrix is deptrecated from numpy, it is done through numpy array which produce exactly the same results and functionality.
# mHat = np.matrix(np.zeros((U, L0)))
# for i in range(0, U):
#     k = np.linalg.inv(BList[i])
#     m = np.dot(k, QList[i])
#     mHat[i, :] = m.T


##
## amended version ##
# M = np.zeros((L,L))
MList= []
for t in range(0, I):
    MList.append([])
    MList[t]= np.dot(mHat[:,1:].T,np.multiply(mHat[:,1:],np.atleast_2d(pTHat[t]).T))*delta
##
# Vector S
## amended ###
S = np.zeros((1,L))
SList = np.tile(S, (I,1,1))
for t in range(0,I):
    S1= sum(np.multiply(np.atleast_2d(qTHat[1]).T, mHat[:,1:]))
#     S2= np.multiply((np.atleast_2d(pTHat1[1]).T),mHat[:,0])
    S2= sum(np.multiply(np.multiply(np.atleast_2d(pTHat[t]).T, mHat[:,0]) , mHat[:,1:]))
    SList[t]= (S1-S2) * delta
##
### amended version###
# updated version with ndarray instead of matrix design.
# Vector ZHat for each date t of length L
ZHat = np.array(np.zeros((I, L)))
for t in range(0, I):
    #     k= np.linalg.inv(MList[t])
    #     m= np.dot(k,SList[t].T)
    ZHat[t] = SList[t].dot(np.linalg.inv(MList[t]))
ZHat = np.concatenate(((np.ones((I, 1))), ZHat), axis=1)
#
# ZHat= np.matrix(np.zeros((I,L)))
# for t in range(0, I):
# #     k= np.linalg.inv(MList[t])
# #     m= np.dot(k,SList[t].T)
#     ZHat[t]= SList[t].dot(  np.linalg.inv(MList[t])  )
# ZHat= np.concatenate( ( (np.ones((I,1))) , ZHat), axis=1)
##
stopCriterionAll= []
for t in range(0,I):
    stopCriterionAll.append([])
    stopCriterionAll[t] = sum(np.multiply(  (np.multiply(ZHat[0],mHat) - np.multiply(ZHatOld[0], mHatOld))    ,   (np.multiply(ZHat[0],mHat) - np.multiply(ZHatOld[0], mHatOld))    )).sum() * delta
stopCriterion=  sum(stopCriterionAll)
ZHatOld= ZHat
mHatOld= mHat
plotStopCriterion.append([])
# plotStopCriterion[it]= stopCriterion   ### to be uncommented with the while loop
##
# Orthogonalization ----------------------------------------------------------- #



## First Step
## pHat
pHat= (1/I) * sum(np.asarray(pTHat), 0) ### equavelant to sum columns of pTHat of the kernel function
## Vector gamma
g=sum((np.multiply(mHat[:,0], np.multiply(mHat[:,1:], np.atleast_2d(pHat).T)))) * delta
## Matrix Gamma
G= np.dot(mHat[:,1:].T  ,  np.multiply(mHat[:,1:], np.atleast_2d(pHat).T) ) * delta
## mHat new
mHat[:,0] = mHat[:,1] - np.dot( np.dot(g,np.linalg.inv(G))   ,  mHat[:,1:].T ).T
mHat[:,1:]=(sp.linalg.fractional_matrix_power(np.linalg.inv(G), 0.5)).T.dot(mHat[:,1:].T).T
## ZHat new
for t in range(0,I):
    ZHat[t,1:]= np.dot(sp.linalg.fractional_matrix_power(G,0.5)  ,   ZHat[1,1:].T + np.dot(np.linalg.inv(G) , g.T)).T

##
# Second Step ------ #
# Matrix B
B= ZHat[:,1:].T.dot(ZHat[:,1:])
# Eigenvectors Z
Z= np.linalg.eig(B)[1]
# mHat new
mHat[:,1:]= Z.T.dot(mHat[:,1:].T).T
# ZHat new
ZHat[:,1:]= Z.T.dot(ZHat[:,1:].T).T
##
# Fit the model ------------------------------------------------------------- #


y= dsfm["iv"]
YBar= y.mean()
YHat= []
for t in range(0, I):
    m_l= []
    for l in range(0,L):
        m_l.append([])
        f= sp.interpolate.SmoothBivariateSpline(u.u2Points,u.u1Points, mHat[:,l])
        m_l[l]= np.matrix(f(x1[t], x2[t]))
    YTHat= []
    for l in range(0,L):
        YTHat.append([])
        YTHat[l] = np.matrix(ZHat[t,l]* m_l[l])
    YHat.append([])
    YHat[t]= functools.reduce(operator.add, YTHat)

YHatRMSE  =  YHat.copy()
YHatList=np.concatenate([np.concatenate(np.array(i)).tolist() for i in YHat]).tolist()
YList= np.concatenate([np.concatenate(np.array(i.T)).tolist() for i in Y]).tolist()
residuals= y - YHatList
numerator= sum([ x**2 for x in residuals])
denominator= sum([(y-YBar)**2  for y in YList])
EV= 1 - (numerator/denominator)
##
# Goodness-of-fit - Root Mean Squared Error - RMSE ---- #
numeratorRMSE = []
for t in range(0, I):
    numeratorRMSE.append([])
    numeratorRMSE[t] = (1 / J[t]) * (Y[t].T - YHatRMSE[t]).T

### standardize the length of all matrices
# max(list(map(len,list(map(np.transpose, numeratorRMSE)))))
for n in range(0, len(numeratorRMSE)):
    # numeratorRMSE[n]= numeratorRMSE[n].T
    while len(numeratorRMSE[n]) < max(map(len, numeratorRMSE)):##max(list(map(len, list(map(np.transpose, numeratorRMSE))))):
    #     # a= np.vstack((numeratorRMSE[n].T, np.zeros((1, len(numeratorRMSE[n])))))
    #     #a = np.vstack((numeratorRMSE[n].T, np.zeros((1, len(numeratorRMSE[n])))))
        numeratorRMSE[n] = np.vstack((numeratorRMSE[n], np.zeros((1,5))))

    # else:
    #     numeratorRMSE[n] = numeratorRMSE[n].T
##
numeratorRMSE = functools.reduce(operator.add, numeratorRMSE)
numeratorRMSE = np.multiply(numeratorRMSE, numeratorRMSE).sum()  ## check if result should be matrix or scalar
RMSE = np.sqrt((1 / I) * numeratorRMSE)
##



def NormalKernel1D(x):
    kernel1D = (1 / np.sqrt(2 * pi)) * np.exp(-0.5 * (x ** 2))
    return kernel1D
##

# Bandwidth Selection  -------------------------------- #
w= 1/pHat
N= len(YList)
kh= ((1/(h[0]*h[1])) * NormalKernel1D(0 / h[0])) * (NormalKernel1D(0/h[1]))
# Weighted AIC_2
wAIC = (1/N) * numerator * np.exp(2 * (L/N) * kh * sum(w) / sum(w* pHat))
# Weighted SC_1
wSC= (1/N) * numerator * np.exp(np.log(N) * (L/N)* kh* sum(w)/ sum(w*pHat))
##
