pTjHatList = []
qTjHatList = []
for t in range(0, I):
    for n in range(0, U):
        o = np.dot(
            (1 / h[1]) * (np.transpose(np.atleast_2d(quartickernel1d((u.u2Points.iloc[n] - x2[1]) / h[1])))),
            np.atleast_2d((1 / h[0]) * quartickernel1d((u.u1Points.iloc[n] - x1[t]) / h[0])
                          )
        )
        pTjHatList.append(o)
        l = pTjHatList[n] * np.asmatrix(y[t])
    ## to be transposed to python
    # p.hat and q.hat function
    pTHat[t,] < - unlist(lapply(pTjHatList, function(x)(1 / J[[t]]) * sum(x)))
    qTHat[t,] < - unlist(lapply(qTjHatList, function(x)(1 / J[[t]]) * sum(x)))
    # Ji*p.hat and Ji*q.hat
    jPTHat[t,] < - unlist(lapply(pTjHatList, function(x)
    sum(x)))
    jQTHat[t,] < - unlist(lapply(qTjHatList, function(x)
    sum(x)))

    list(pHat=pTHat, qHat=qTHat, jPHat=jPTHat, jQHat=jQTHat)
