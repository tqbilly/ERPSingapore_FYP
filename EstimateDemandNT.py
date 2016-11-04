import json
import numpy as np
import math
#from igraph import *
import os
import find_shortestp
import allpairshortestp
from itertools import chain

# Given the characteristics of the dataset, we are defining free-flow speed as the 95% - percentile, instead of 85%

# speed: KM/H
# order: AD BA BE CB DE DG EB EF EH FC GH HE HI IF
#        1  2  3  4  5  6  7  8  9  10 11 12 13 14

# calculate the avearge speed & freespeed on each road
file = open('/Users/billy/Desktop/FYP/NT/traffic_speedbandBANT.json','r')
maxBA=[]
minBA=[]
speedBA=[]
try:
    count = 0
    while True:
        count=count+1
        line=file.readline()
        if line:
            if count%8==4:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                maxBA.append(d["MaximumSpeed"])
            if count%8==5:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                minBA.append(d["MinimumSpeed"])
        else:
            break
finally:
    file.close()
lenBA=len(maxBA)

file = open('/Users/billy/Desktop/FYP/NT/traffic_speedbandADNT.json','r')
maxAD=[]
minAD=[]
speedAD=[]
try:
    count = 0
    while True:
        count=count+1
        line=file.readline()
        if line:
            if count%8==4:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                maxAD.append(d["MaximumSpeed"])
            if count%8==5:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                minAD.append(d["MinimumSpeed"])
        else:
            break
finally:
    file.close()
lenAD=len(maxAD)

file = open('/Users/billy/Desktop/FYP/NT/traffic_speedbandBENT.json','r')
maxBE=[]
minBE=[]
speedBE=[]
try:
    count = 0
    while True:
        count=count+1
        line=file.readline()
        if line:
            if count%8==4:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                maxBE.append(d["MaximumSpeed"])
            if count%8==5:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                minBE.append(d["MinimumSpeed"])
        else:
            break
finally:
    file.close()
lenBE=len(maxBE)

file = open('/Users/billy/Desktop/FYP/NT/traffic_speedbandCBNT.json','r')
maxCB=[]
minCB=[]
speedCB=[]
try:
    count = 0
    while True:
        count=count+1
        line=file.readline()
        if line:
            if count%8==4:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                maxCB.append(d["MaximumSpeed"])
            if count%8==5:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                minCB.append(d["MinimumSpeed"])
        else:
            break
finally:
    file.close()
lenCB=len(maxCB)

file = open('/Users/billy/Desktop/FYP/NT/traffic_speedbandDENT.json','r')
maxDE=[]
minDE=[]
speedDE=[]
try:
    count = 0
    while True:
        count=count+1
        line=file.readline()
        if line:
            if count%8==4:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                maxDE.append(d["MaximumSpeed"])
            if count%8==5:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                minDE.append(d["MinimumSpeed"])
        else:
            break
finally:
    file.close()
lenDE=len(maxDE)

file = open('/Users/billy/Desktop/FYP/NT/traffic_speedbandDGNT.json','r')
maxDG=[]
minDG=[]
speedDG=[]
try:
    count = 0
    while True:
        count=count+1
        line=file.readline()
        if line:
            if count%8==4:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                maxDG.append(d["MaximumSpeed"])
            if count%8==5:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                minDG.append(d["MinimumSpeed"])
        else:
            break
finally:
    file.close()
lenDG=len(maxDG)

file = open('/Users/billy/Desktop/FYP/NT/traffic_speedbandEBNT.json','r')
maxEB=[]
minEB=[]
speedEB=[]
try:
    count = 0
    while True:
        count=count+1
        line=file.readline()
        if line:
            if count%8==4:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                maxEB.append(d["MaximumSpeed"])
            if count%8==5:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                minEB.append(d["MinimumSpeed"])
        else:
            break
finally:
    file.close()
lenEB=len(maxEB)

file = open('/Users/billy/Desktop/FYP/NT/traffic_speedbandEFNT.json','r')
maxEF=[]
minEF=[]
speedEF=[]
try:
    count = 0
    while True:
        count=count+1
        line=file.readline()
        if line:
            if count%8==4:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                maxEF.append(d["MaximumSpeed"])
            if count%8==5:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                minEF.append(d["MinimumSpeed"])
        else:
            break
finally:
    file.close()
lenEF=len(maxEF)

file = open('/Users/billy/Desktop/FYP/NT/traffic_speedbandEHNT.json','r')
maxEH=[]
minEH=[]
speedEH=[]
try:
    count = 0
    while True:
        count=count+1
        line=file.readline()
        if line:
            if count%8==4:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                maxEH.append(d["MaximumSpeed"])
            if count%8==5:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                minEH.append(d["MinimumSpeed"])
        else:
            break
finally:
    file.close()
lenEH=len(maxEH)

file = open('/Users/billy/Desktop/FYP/NT/traffic_speedbandFCNT.json','r')
maxFC=[]
minFC=[]
speedFC=[]
try:
    count = 0
    while True:
        count=count+1
        line=file.readline()
        if line:
            if count%8==4:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                maxFC.append(d["MaximumSpeed"])
            if count%8==5:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                minFC.append(d["MinimumSpeed"])
        else:
            break
finally:
    file.close()
lenFC=len(maxFC)

file = open('/Users/billy/Desktop/FYP/NT/traffic_speedbandGHNT.json','r')
maxGH=[]
minGH=[]
speedGH=[]
try:
    count = 0
    while True:
        count=count+1
        line=file.readline()
        if line:
            if count%8==4:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                maxGH.append(d["MaximumSpeed"])
            if count%8==5:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                minGH.append(d["MinimumSpeed"])
        else:
            break
finally:
    file.close()
lenGH=len(maxGH)

file = open('/Users/billy/Desktop/FYP/NT/traffic_speedbandHENT.json','r')
maxHE=[]
minHE=[]
speedHE=[]
try:
    count = 0
    while True:
        count=count+1
        line=file.readline()
        if line:
            if count%8==4:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                maxHE.append(d["MaximumSpeed"])
            if count%8==5:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                minHE.append(d["MinimumSpeed"])
        else:
            break
finally:
    file.close()
lenHE=len(maxHE)

file = open('/Users/billy/Desktop/FYP/NT/traffic_speedbandHINT.json','r')
maxHI=[]
minHI=[]
speedHI=[]
try:
    count = 0
    while True:
        count=count+1
        line=file.readline()
        if line:
            if count%8==4:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                maxHI.append(d["MaximumSpeed"])
            if count%8==5:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                minHI.append(d["MinimumSpeed"])
        else:
            break
finally:
    file.close()
lenHI=len(maxHI)

file = open('/Users/billy/Desktop/FYP/NT/traffic_speedbandIFNT.json','r')
maxIF=[]
minIF=[]
speedIF=[]
try:
    count = 0
    while True:
        count=count+1
        line=file.readline()
        if line:
            if count%8==4:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                maxIF.append(d["MaximumSpeed"])
            if count%8==5:
                str=line
                str=str[:-3]
                str="{"+str+"}"
                d=json.loads(str)
                minIF.append(d["MinimumSpeed"])
        else:
            break
finally:
    file.close()
lenIF=len(maxIF)

xLen=min(lenAD,lenBA,lenBE,lenCB,lenDE,lenDG,lenEB,lenEF,lenEH,lenFC,lenGH,lenHE,lenHI,lenIF)
# print xLen
freespeed=[]
for i in range(0,xLen):
    temp=(maxBA[i]+minBA[i])/2
    speedBA.append(temp)
avspeedBA=np.mean(speedBA)
freespeedBA=np.percentile(speedBA,90)
freespeed.append(freespeedBA)
for i in range(0,xLen):
    temp=(maxAD[i]+minAD[i])/2
    speedAD.append(temp)
avspeedAD=np.mean(speedAD)
freespeedAD=np.percentile(speedAD,90)
freespeed.append(freespeedAD)
for i in range(0,xLen):
    temp=(maxBE[i]+minBE[i])/2
    speedBE.append(temp)
avspeedBE=np.mean(speedBE)
freespeedBE=np.percentile(speedBE,90)
freespeed.append(freespeedBE)
for i in range(0,xLen):
    temp=(maxCB[i]+minCB[i])/2
    speedCB.append(temp)
avspeedCB=np.mean(speedCB)
freespeedCB=np.percentile(speedCB,90)
freespeed.append(freespeedCB)
for i in range(0,xLen):
    temp=(maxDE[i]+minDE[i])/2
    speedDE.append(temp)
avspeedDE=np.mean(speedDE)
freespeedDE=np.percentile(speedDE,90)
freespeed.append(freespeedDE)
for i in range(0,xLen):
    temp=(maxDG[i]+minDG[i])/2
    speedDG.append(temp)
avspeedDG=np.mean(speedDG)
freespeedDG=np.percentile(speedDG,90)
freespeed.append(freespeedDG)
for i in range(0,xLen):
    temp=(maxEB[i]+minEB[i])/2
    speedEB.append(temp)
avspeedEB=np.mean(speedEB)
freespeedEB=np.percentile(speedEB,90)
freespeed.append(freespeedEB)
for i in range(0,xLen):
    temp=(maxEF[i]+minEF[i])/2
    speedEF.append(temp)
avspeedEF=np.mean(sorted(speedEF))
freespeedEF=np.percentile(speedEF,90)
freespeed.append(freespeedEF)
for i in range(0,xLen):
    temp=(maxEH[i]+minEH[i])/2
    speedEH.append(temp)
avspeedEH=np.mean(speedEH)
freespeedEH=np.percentile(speedEH,90)
freespeed.append(freespeedEH)
for i in range(0,xLen):
    temp=(maxFC[i]+minFC[i])/2
    speedFC.append(temp)
avspeedFC=np.mean(speedFC)
freespeedFC=np.percentile(speedFC,90)
freespeed.append(freespeedFC)
for i in range(0,xLen):
    temp=(maxGH[i]+minGH[i])/2
    speedGH.append(temp)
avspeedGH=np.mean(speedGH)
freespeedGH=np.percentile(speedGH,90)
freespeed.append(freespeedGH)
for i in range(0,xLen):
    temp=(maxHE[i]+minHE[i])/2
    speedHE.append(temp)
avspeedHE=np.mean(speedHE)
freespeedHE=np.percentile(speedHE,90)
freespeed.append(freespeedHE)
for i in range(0,xLen):
    temp=(maxHI[i]+minHI[i])/2
    speedHI.append(temp)
avspeedHI=np.mean(speedHI)
freespeedHI=np.percentile(speedHI,90)
freespeed.append(freespeedHI)
for i in range(0,xLen):
    temp=(maxIF[i]+minIF[i])/2
    speedIF.append(temp)
avspeedIF=np.mean(speedIF)
freespeedIF=np.percentile(speedIF,90)
freespeed.append(freespeedIF)
# set up N

# order: AD BA BE CB DE DG EB EF EH FC GH HE HI IF
# proposed matrix:
# 1 1 0 0 0 0 0 0 0 0 0 0 0 0
# 0 1 1 1 0 0 1 0 0 0 0 0 0 0
# 0 0 0 1 0 0 0 0 0 1 0 0 0 0
# 1 0 0 0 1 1 0 0 0 0 0 0 0 0
# 0 0 1 0 1 0 1 1 1 0 0 1 0 0
# 0 0 0 0 0 0 0 1 0 1 0 0 0 1
# 0 0 0 0 0 1 0 0 0 0 1 0 0 0
# 0 0 0 0 0 0 0 0 1 0 1 1 1 0
# 0 0 0 0 0 0 0 0 0 0 0 0 1 1
N=np.array([[-1,1,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,-1,-1,1,0,0,1,0,0,0,0,0,0,0],
            [0,0,0,-1,0,0,0,0,0,1,0,0,0,0],
            [1,0,0,0,-1,-1,0,0,0,0,0,0,0,0],
            [0,0,1,0,1,0,-1,-1,-1,0,0,1,0,0],
            [0,0,0,0,0,0,0,1,0,-1,0,0,0,1],
            [0,0,0,0,0,1,0,0,0,0,-1,0,0,0],
            [0,0,0,0,0,0,0,0,1,0,1,-1,-1,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,-1]])
N=np.matrix(N)

# set up speed vector
#speedAD = sorted(speedAD)
#speedBA = sorted(speedBA)
#speedBE = sorted(speedBE)
#speedCB = sorted(speedCB)
#speedDE = sorted(speedDE)
#speedDG = sorted(speedDG)
#speedEB = sorted(speedEB)
#speedEF = sorted(speedEF)
#speedEH = sorted(speedEH)
#speedFC = sorted(speedFC)
#speedGH = sorted(speedGH)
#speedHE = sorted(speedHE)
#speedHI = sorted(speedHI)
#speedIF = sorted(speedIF)
speed = []
bigV = []
for i in range(0, xLen):
    bigV = [speedAD[i], speedBA[i], speedBE[i], speedCB[i], speedDE[i], speedDG[i], speedEB[i], speedEF[i],
            speedEH[i], speedFC[i], speedGH[i], speedHE[i], speedHI[i], speedIF[i]]
    speed.append(np.transpose(bigV))
temp = speed[0:420]#330 - not correct
speed = temp
#temp = [0.0 for k in range(14)]
#for i in range(0,14):
#    temp[i]=freespeed[i]
#speed=[]
#speed.append(np.transpose(temp))
#speed.append(np.transpose(temp))
#speed.append(np.transpose(temp))
#print temp
#print type(speed)
lenspeed = len(speed)
xLen = len(speed)



# set up flow capacity - we estimate the capacity using outside source for calculation, since we cannot get the data from LTA
# http://midimagic.sgc-hosting.com/capaz.htm
# parameter used:
# 1. numOfLanes+lane type:
#    AD: Cantonment Rd, 2 thru, 1 thru+left, 1 thru+right - 5083 vh/hr
#    BA: Hoe Chiang Rd, 3 left - 3951 vh/hr
#    BE: Tg Pagar Rd, 1 thru+left, 1 thru - 2512 vh/hr
#    CB: Enggor Rd, 1 thru, 1 thru+left(can park), 1 thru+right - 3268 vh/hr
#    DE: Lim Teck Kim Rd, 1 thru+left, 1 left, 1 thru+right - 3768 vh/hr
#    DG: Cantonment Rd, 1 thru, 1 thru+left, 1 thru+right, 1 right - 5024 vh/hr
#    EB: Tg Pagar Rd, 1 thru+left, 1 thru - 2512 vh/hr
#    EF: Bernam Rd, 1 left, 1 left(can park), 1 thru+left(can park) - 3327 vh/hr
#    EH: Tg Pagar Rd, 2 left - 2571 vh/hr
#    FC: Anson Rd, 3 thru, 1 thru+left - 5207 vh/hr
#    GH: Keppel Rd, 2 thru, 1 left, 1 thru+right - 5083 vh/hr
#    HE: Tg Pagar Rd, 1 thru, 1 thru+right - 2512 vh/hr
#    HI: Keppel Rd, 2 thru, 1 left, 1 right - 5083 vh/hr
#    IF: Anson Rd, 3 thru, 1 thru+left - 5207 vh/hr
# 2. laneWidth in Singapore: 290cm(average lane width in Singapore) , source: http://lovecycling.net/projects/lane-width/
# 3. 95-percentile speed(freespeed)
# 4. heavy vehicle portion: 5%
# 5. signal green portion: 60%
m=[5083,3951,2512,3268,3768,5024,2512,3327,2571,5207,5083,2512,5083,5207]
# set up z
z=[]
#minArc=[min(speedAD),min(speedBA),min(speedBE),min(speedCB),min(speedDE),min(speedDG),min(speedEB),min(speedEF),min(speedEH),min(speedFC),min(speedGH),min(speedHE),min(speedHI),min(speedIF)]
for i in range(0,xLen):
    xtemp=[]
    for j in range(0,14):
        #temp=4*m[j]*(speed[i][j]/freespeed[j])-16*m[j]*m[j]*((speed[i][j]/freespeed[j])**2) #Greensheild Method
        # change to modified Greensheild Method? - Underwood Exponential Model
        #temp=freespeed[j]*math.exp(-((minArc[j]/freespeed[j])**2)/((speed[i][j]/freespeed[j])**2))*(speed[i][j]/freespeed[j])*m[j]
        # change to another model? - Pipe's Generalized Model
        kj=4*m[j]/freespeed[j]
        #temp=((speed[i][j]*kj)**2-(speed[i][j]**3)*(kj**2)/freespeed[j])**(1/2)
        temp=(kj*speed[i][j]/(math.exp(speed[i][j]/freespeed[j]))) #- best of all: Underwood Exponential Model
        if temp<0:
            temp=0
        xtemp.append(temp)
    z.append(np.transpose(xtemp))
#print z
lenZ=len(z)

# set up free-flow travel time using the freespeed data
#AD  0.0175
#1.2740556077200536 103.84137702698492 1.2739131248065627 103.84144393152431
#BA 0.1909
#1.2741031259190334 103.84314902985783 1.2739131248065627 103.84144393152431
#BE 0.0667
#1.2741031259190334 103.84314902985783 1.2735075470714696 103.84321380304505
#CB 0.1683
#1.2742629545997886 103.84522124314651 1.2741748488866387 103.84371150465546
#DE 0.172
#1.2734704645565817 103.84166906725088 1.2735075470714696 103.84321380304505
#DG 0.0535
#1.2734704645565817 103.84166906725088 1.2730632100738564 103.84192440279857
#EB 0.0667
#1.2741031259190334 103.84314902985783 1.2735075470714696 103.84321380304505
#EF 0.158
#1.2735075470714696 103.84321380304505 1.2735468810789548 103.84463325604787
#EH 0.0667
#1.2735075470714696 103.84321380304505 1.2741031259190334 103.84314902985783
#FC 0.0631, 0.1038
#1.2742629545997886 103.84522124314651 1.2746957819444513 103.84558747048918
#1.273480, 103.844628 1.274193, 103.845229
#GH 0.0599, 0.1171
#1.2726934746440954 103.84301633779586 1.2727139835543118 103.84355397656213
#1.272785, 103.842190 1.272794, 103.843242
#HE 0.0667
#1.2735075470714696 103.84321380304505 1.2741031259190334 103.84314902985783
#HI 0.0144, 0.0915
#1.2727139835543118 103.84355397656213 1.2726405577356923 103.84366017417545
#1.272794, 103.843242 1.272823, 103.844064
#IF 0.1142
#1.272737232949127 103.8440027322195 1.2735468810789548 103.84463325604787
RoadLen=[0.0175,0.1909,0.0667,0.1683,0.172,0.0535,0.0667,0.158,0.0667,0.1038,0.1171,0.0667,0.0915,0.1142]
t0=[]
for i in range(0,14):
    t0.append(RoadLen[i]/freespeed[i])

# set up arc flow - parameter e
e=[]
for i in range(0,14):
    eArc=[]
    for j in range(0,lenZ):
        eArc.append(z[j][i])
    e.append(np.transpose(eArc))

# set up the demand dw - GLS method
mygraph=allpairshortestp.mygraph
h=allpairshortestp.h

# Transportation Systems Analysis: Models and Applications - definition of path-link incidence matrix
# Assume all drivers are choosing the shortest path: we get the path-link incidence matrix A

Alist=find_shortestp.plm
A=np.matrix(Alist)
At=np.transpose(A)

# set up p - prob. matrix
# logit parameter: 0.5
#p = [ [ 0 for i in range(14) ] for j in range(14) ]
#for i in range(0,14):
#    temp=0
#    for j1 in range(0,14):
#        temp=temp+math.exp(-0.5*h[i][j1])
#    for j2 in range(0,9):
#        p[i][j2]=(math.exp(-0.5*h[i][j2]))/temp

#p=np.matrix(p)
#pt=np.transpose(p)
h=np.matrix(h)
mygraph=np.matrix(mygraph)
# we are assuming that all drivers are choosing the shortest path - selfish user in a congestion game

# S - sample covariance matrix
zlist=z
z=np.matrix(z)
zt=np.transpose(z)
S=np.cov(zt)
if np.linalg.det(S)==0:
    Sinv=np.linalg.pinv(S).real #if S is Singular, then we use pseudo-inverse: are all sample covariance singular?
else:
    Sinv=np.linalg.inv(S).real

#set up Q
Q=np.dot(A,np.dot(Sinv,At))

#set up b
b=[0 for n in range(0,14)]
for k in range(0,lenZ):
    btemp=[0 for n in range(0,14)]
    for i in range(0,14):
        btemp[i]=zlist[k][i]
    b=[x+y for x,y in zip(b,btemp)]
b=np.matrix(b)
bt=np.transpose(b)
b=np.dot(A,np.dot(Sinv,bt))


# perform QP to find O-D Demand Information
from cvxopt import matrix
from cvxopt import solvers
P = matrix(Q)*lenZ
q = -matrix(b)
G = matrix(-np.identity(72))
h = [0.0 for i in range(72)]
h = matrix(h)


sol = solvers.qp(P,q,G,h)
d=sol['x']

flowt = [0 for i in range(0, 14)]
for j in range(14):
    flowt[j] = np.mean(e[j])
