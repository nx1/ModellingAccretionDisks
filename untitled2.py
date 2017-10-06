# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 13:28:25 2017

@author: yv
"""

def createdisc (N,const,Rmax,rstart):
    R=[]
    for i in range(N):
        if i==0:
            R.append(rstart)
        else:
            R.append(R[i-1]*const)
    print R
       
createdisc(100,3,8,2)
