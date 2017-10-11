# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 13:28:25 2017

@author: yv
"""
import numpy as np

def create_disc (N,const,Rmax,rstart):
    R=[]
    for i in range(N):
        if i==0:
            R.append(rstart)
        else:
            R.append(R[i-1]*const)
    return np.asarray(R)        #Converts python list to numpy array
       
print(create_disc(10,3,8,2))
