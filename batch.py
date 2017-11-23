# -*- coding: utf-8 -*-




#execfile("discfunc.py")


N = [1,5,10,30]
const = [1.1,1.5]
Q_factor = [0.005,0.025,0.1]
tMax_factor = [1.1,10.0]
H_R_= [0.1,1.0]
M_0_start = [1.0,10.0]

map1 = map(None, N, const)

for x, y in map1:
    print x, y