import numpy as np

a=[1,1,1,1,1]

for i in [1,2]:
    for j in [1,2]:
        for k in [1,2]:
            a = [1, 1, 1, 1, 1]
            a[1] = a[1] * ((-1) ** i)
            a[2] = a[2] * ((-1) ** j)
            a[3] = a[3] * ((-1) ** k)
            res = 0
            for t in range(0, 4):
                res += a[t] * a[t + 1]
            print res
            #print a

