# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 13:28:25 2017

@author: yv
"""
import numpy as np
import matplotlib.pyplot as plt


def create_disc (N,const,Rmax,rstart):
    R=[]
    for i in range(N):
        if i==0:
            R.append(rstart)
        else:
            R.append(R[i-1]*const)
    return np.asarray(R)        #Converts python list to numpy array
       

def tempeff (R):
    temp=[]
    for i in range(len(R)):
        temp.append(R[i]**(-0.75))
    return temp
    
    
R=create_disc(10,3,8,2)
t=tempeff(R)

ax = plt.subplot(111)
ax.set_xscale("log", nonposx='clip')
ax.plot(R,t)