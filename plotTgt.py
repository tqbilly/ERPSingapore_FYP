import matplotlib.pyplot as plt
import numpy as np
import InverseVIAM
import InverseVINT

res1 = InverseVIAM.myres
res2 = InverseVINT.myres

erp = InverseVIAM.erp_choice
ma = InverseVIAM.maxFlow
t0 = InverseVIAM.t0
x=np.linspace(0,1,30)
y1=[]
y2=[]
y3=[]
y4=[]
for i in range(0,30):
    temp1=0
    temp2=0
    temp3=0
    temp4=0
    for j in range(0,5):
        temp1+=res1[j]*(x[i]**j)*t0[0]
        temp2+=res2[j]*(x[i]**j)*t0[0]
        #temp3+=res1[j]*(x[i]**j)
        temp4+=(1+j)*res1[j]*(x[i]**j)*t0[0]
    y1.append(temp1)
    y2.append(temp2)
    y3.append(temp1+erp*t0[0])
    y4.append(temp4)
plt.xlabel('xa/ma')
plt.ylabel('scaled cost')
plt.plot(x,y1,"r--",x,y2,'bs',x,y4,'y-')
plt.show()