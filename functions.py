print("Hello World")
import numpy as np
import time
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import itertools

np.random(3,2,1)
import pandas as pd
import matplotlib.pyplot as plt


def DSFM2D(data, numDataPoints= 25, h=[0.5,0.5], L=3, initialLoad= "WN",tol = 1e-5, maxIt = 301):
    Time = time.clock()

    # initial settings:
    date = np.unique(dsfm["data_date"])
    I    = len(date)
    L0   = L+1
    Y   = []
    x1   = []
    x2   = []
    J = []
    for i in range(0, I):
        z1 = np.array(np.unique(dsfm.loc[dsfm["data_date"] == date[i]]["TTM"]))
        z2 = np.array(np.unique(dsfm.loc[dsfm["data_date"] == date[i]]["LM"]))
        y = np.array(np.unique(dsfm.loc[dsfm["data_date"] == date[i]]["iv"]))
        j = len(z1) * len(z2)
        print(J)
        x1.append(z1)
        x2.append(z2)
        J.append(j)
        Y.append(y)
        ###>>>>> alternative structure <<<< ####
        #Y=[]
        # for i in range(0,I):
        #     y= np.array(np.unique(dsfm.loc[dsfm["data_date"] == date[10]]["iv"]))
        #     y=np.matrix(y.reshape(len(x1[i]),len(x2[i])))
        #     Y.append((y))
        # Y=np.asarray(Y)
        #### >>>>> for y to be a matrix #######
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    J = np.asarray(J)
    Y = np.asarray(Y)

    for i in range(0, len(x1)):### lapply(x1, function(x) x / max(x))
        x1[i] = x1[i] / x1[i].max()
    for i in range(0, len(x2)):
        x2[i] = x2[i] / x2[i].max()

    # initual loadings Z_t,j
    #def initload(initialloads):
    #AR=  ZHat
    alphas = np.array([0.4])
    betas = np.array([0.])
    ar = np.r_[1, alphas]
    ma = np.r_[1, betas]
    ar1 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=I)
    ZHat = []
    for i in range(0, L):
        a = smt.arma_generate_sample(ar=ar, ma=ma, nsample=I)
        ZHat.append(a)
    ZHat = np.array(ZHat)

    ZHat = np.c_[(np.ones((I, 1)), ZHat.T)]

    # Create a regular grid of points u covering the whole space

    minx1 = np.min([y for x in x1 for y in x])
    minx2 = np.min([y for x in x2 for y in x])
    maxx1 = np.max([y for x in x1 for y in x])
    maxx2 = np.max([y for x in x2 for y in x])
    delta = ((maxx1 - minx1) * (maxx2 - minx2)) / numDataPoints ** 2
    u1 = np.linspace(minx1, maxx1, num=numDataPoints)
    u1Points = np.linspace(minx1, maxx1, num=numDataPoints)
    u2 = np.linspace(minx2, maxx2, num=numDataPoints)
    u2Points = np.linspace(minx2, maxx2, num=numDataPoints)
    u1Points = np.tile(u1Points, len(u2Points))
    u1Points = np.sort(u1Points)
    u2Points = np.tile(u2Points, len(u2Points))
    u = pd.DataFrame({'u1Points': u1Points, 'u2Points': u2Points})
    U = len(u1) * len(u2)

    # Loop parameters
    ZHatOld=np.zeros((I,L0))
    mHatOld=np.zeros((U,L0))
    it=0
    stopCriterion=1
    plotStopCriterion=

    # KERNEL -------------------------------------------------------------------- #
    # 2 Dimensional Kernel
    h = np.tile(h, (numDataPoints * numDataPoints, 1))
    # kernel function to be transformed later
    #Kernel < - KernelDensity2D(y, I, J, x1, x2, u, U, h)

    # LOOP ---------------------------------------------------------------------- #

    # Matrix B  #### TO BE EDITED ####
    Bu = np.zeros((1, L0, L0))
    # Bu=np.matrix(Bu)
    BList = np.tile(Bu, (U, 1, 1))
    for i in range(0, U):
        BList[i] = np.dot(ZHat.T, (ZHat + i))
    # Vector Q  #### TO BE EDITED ####
    Q = np.zeros((1, L0))
    QList = np.tile(Q, (U, 1))
    for i in range(0, U):
        QList[i] = np.dot(ZHat.T, np.ones((11, 1))).T

    # Vector mHat for each grid points of length L
    mHat = []
    for i in range(0, U):
        c[i]= np.linalg.inv(BList[i])
        m = np.dot(c[i], QList[i].T)
        mHat.append(m)
    # Matrix M  **** to be edited after creating the Kernel function
    M = np.zeros((1, L0, L0))
    MList = np.tile(M, (I, 1, 1))
    mHat = np.array(mHat)
    for i in range(0, I):
        MList[t] = np.dot(mHat[:, 1:], mHat[:, 1:].T * Kernel.pHat[t,]) * delta

    # Vector S
    S = np.zeros((1, L))
    SList = np.tile(S, (I, 1, 1))
    for i in range(0, I):
    # s1= pd.DataFrame(mHat[:,1:]).multiply(pd.Series(np.random.rand(625)), axis=0)
    ### to be continued #####

    # Vector ZHat for each date t of length L
    ZHat = []
    for i in range(0, U):
        c[i] = np.linalg.inv(MList[i])
        m = np.dot(c[i], SList[i].T)
        ZHat.append(m)
    ZHat= np.array(ZHat)
    ### add following line ####
    ###ZHat <- cbind(1, ZHat)###

    # Stopping Criterion #### to be changed for python ####
    stopCriterionAll < - vector(mode="numeric", I)
    for (i in 1:I) {
        stopCriterionAll[[t]] < - sum((rowSums(ZHat[t,] * mHat - ZHatOld[t,] *
                                               mHatOld) ^ 2)) * delta
    }
        stopCriterion < - sum(stopCriterionAll)

    # Orthogonalization --------------------------------------------------------- #
    # Orthogonalization --------------------------------------------------------- #
    # First Step ------- #
    # First Step ------- #
    # pHat




