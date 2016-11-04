import json
import numpy as np
import math
from cvxopt import matrix
from cvxopt import solvers
import scipy.optimize
import sympy

import find_shortestp
import allpairshortestp
import EstimateDemandNT

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

xLen=EstimateDemandNT.xLen
freespeed=EstimateDemandNT.freespeed
speed=EstimateDemandNT.speed

# parameter N: 9*14
N=EstimateDemandNT.N

# parameter A: 72*14
plm=find_shortestp.plm
Aqp=[]
for i in range(0,72):
    temp=[0.0 for n in range(0,14)]
    for j in range(0,14):
        temp[j]=plm[i][j]
    Aqp.append(temp)
Aqp=np.matrix(Aqp)

# parameter e: 14*1
e = EstimateDemandNT.e
flow = [0 for i in range(0, 14)]
for j in range(14):
    flow[j] = np.mean(e[j])
compa = [0 for i in range(0, 14)]
flowtemp=flow
for j in range(14):
    compa[j] = flow[j] / freespeed[j]
# parameter z: 14*1

# parameter ta0: 14*1
t0=EstimateDemandNT.t0

# parameter d: 9*8
d=EstimateDemandNT.d
d=np.matrix(d)
dmat=d.reshape(9,8)

t=[]
for i in range(0,72):
    temp = [0.0 for h in range(0,14)]
    for j in range(0,14):
        temp[j] = Aqp[i].item(j)*d.item(i)
    t.append(temp)
dw = []
for i in range(0,72):
    temp = np.dot(N, np.transpose(t[i]))
    dw.append(temp)

# set a function to find inverse
def finv(A):
    res=A
    if np.linalg.det(A)==0:
        res=np.linalg.pinv(A)
    else:
        res=np.linalg.inv(A)
    return res

da = np.dot(np.transpose(d),plm)
m = EstimateDemandNT.m
k = []
for i in range(0,14):
    k.append(4*m[i]/freespeed[i])
vdemand = []
for i in range(0,14):
    temp = scipy.optimize.fsolve(lambda x: k[i]*x*math.exp(-x/freespeed[i])-flowtemp[i], 0)
    vdemand.append(temp)

# normalize flow
sq = 0
for i in range(0,14):
    sq += flow[i]**2
sq = math.sqrt(sq)
for i in range(0,14):
    flow[i] = flow[i]/sq


# parameter e: 14*1

zlist = EstimateDemandNT.zlist
t1 = EstimateDemandNT.flowt
m = EstimateDemandNT.m

e = EstimateDemandNT.e
totalLen = len(e[0])
maxFlow = []
for i in range(0,14):
    maxFlow.append(max(e[i]))

def get_elist(start,stop):
    res=[]
    for i in range(0,14):
        temp=[]
        for j in range(start,stop):
            tmp = (int(int(e[i][j])/maxFlow[i]*1e6))/1e6
            temp.append(tmp)
        res.append(np.transpose(temp))
    return res


ea1 = np.identity(14)
ea1 = np.matrix(ea1)
ea = []
for i in range(0, 14):
    ea.append(ea1[i])

# set up QP using cvxopt
def InvVI(c,deg,lamb,erp,el):


    noNode=9
    noPair=noNode*(noNode-1) #72
    noLink=14

    for i in range(noLink):
        t0[i]=t0[i]

    flow=[]
    for i in range(14):
        temp = np.mean(el[i])
        flow.append(temp)
    lendeg=deg+1
    # constraint 1
    con1=[[0.0 for n in range(lendeg+noNode*noPair+1)]for k in range(noLink*noPair)]
    for a in range(0,noLink):
        for w in range(0,noPair):
            for i in range(0,noPair):
                for j in range(0,lendeg):
                    con1[i+noPair*a][j] = -t0[a]*((flow[a])**j)+erp
                for v in range(0,noNode):
                    Ntemp = np.transpose(N)
                    con1[i+noNode*a][lendeg+noNode*i+v] = Ntemp[a].item(v)

    # constraint 2
    con2 = []
    val2 = []
    for i in range(0, noLink):
        for j in range(noLink):
            contemp = [0.0 for n in range(0, lendeg+noNode*noPair+1)]
            if (i != j and flow[i] <= flow[j]):
                for k in range(0, lendeg):
                    contemp[k] = flow[i]**k-flow[j]**k
                con2.append(np.transpose(contemp))
    con2 = np.matrix(con2)

    # constraint 3
    con3 = [0.0 for n in range(lendeg+noNode*noPair+1)]
    for i in range(0,lendeg):
        temp=0
        for a in range(0,noLink):
            temp += t0[a]*(flow[a]*m[a])*(flow[a]**i)+erp
        con3[i] = temp
    for w in range(0,noPair):
        for k in range(0,noNode):
            con3[lendeg+k*w]=dw[w].item(k)
    con3[len(con3)-1]=-1
    con3=np.matrix(con3)

    # additional constraint on y
    con = np.identity(lendeg+noNode*noPair+1)
    for i in range(0,lendeg):
        con[i][i]=0
    con[lendeg+noNode*noPair][lendeg+noNode*noPair]=0


    # constraint 4
    con4 = [0.0 for n in range(lendeg+noNode*noPair+1)]
    con4[0] = 1
    con4 = np.matrix(con4)

    kt = np.concatenate((con1, con2), axis=0)
    kt = np.concatenate((kt, con3), axis=0)
    kt = np.concatenate((kt, con), axis=0)

    # perform QP
    P = [[0.0 for n in range(lendeg+noNode*noPair+1)] for k in range(lendeg+noNode*noPair+1)]
    for i in range(0,lendeg):
        P[i][i] = 1/(nCr(deg,i)*(c**(deg-i)))*2
    P[lendeg+noNode*noPair][lendeg+noNode*noPair]=lamb
    q = matrix([0.0 for n in range(0, lendeg+noNode*noPair+1)])
    P = matrix(P)
    q = matrix(q)
    G = matrix(kt)
    h = matrix([0.0 for n in range(np.shape(kt)[0])])
    A = matrix(con4)
    b = [1.0+erp]
    b = matrix(b)

    sol = solvers.qp(P, q, G, h, A, b)
    res = sol['x']

    return res

myres = InvVI(1.2,5,100,0,get_elist(0,300))# best key: 1.2, 5, 100
import matplotlib.pyplot as plt
x=np.linspace(0,1,30)
y=[]
for i in range(0,30):
    temp=0
    for j in range(0,5):
        temp+=myres[j]*(x[i]**j)
    y.append(temp)
plt.xlabel('xa/ma')
plt.ylabel('scaled cost')
plt.plot(x,y,'r--')
plt.show()