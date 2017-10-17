# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:44:25 2017

@author: yv
"""
from math import *
import random

def create_disc (N,const,Rmax,rstart):
    
    '''
    Creates a basic disc with constant ratio between radii
    Returns a numpy array.
    
    inputs:
        N       = Number of Annuli
        const   = Constant ratio between consecutive radii
        Rmax    = Max Radius (CURRENTLY UNUSED)
        rstart  = Radii of innermost disc
    '''
    
    R = []
    for i in range(N):
        if i==0:
            R.append(rstart)
        else:
            R.append(R[i-1]*const)
            
    return R    #Converts python list to numpy array
          
        
def M_dot(R, t, M_0_Start):

    '''
    Calculates the mass accretion at a given radii between two annuli 
    
    inputs:
        R    = Array of the radii
        t    = time (CURRENTLY UNUSED)
    Dont think this is right but whatevs
    '''
    
    print('INPUTS:', R)
    M_dot = [0.0]*len(R)
    m_dot_local = [0.0]*len(R)
    M_dot_local = [0.0]*len(R)
    M_dot_local[len(R)-1] = M_0_Start
    multm_dot=1.0 #multiplicative term 
    
    for i in range(len(R)):# creating m_dot_local
        m_dot_local[i]=random.uniform(0.0,0.1)
        #m_dot_local is <<1
    m_dot_local.sort()
    m_dot_local.reverse()
    print ("mdot in descending order",m_dot_local)
    
    for i in range(len(R)-1,-1,-1):
        if i==(len(R)-1):
            M_dot[i]=M_dot_local[i]*(m_dot_local[i]+1)
            multm_dot=multm_dot*(m_dot_local[i]+1)
        else:
            multm_dot=multm_dot*(m_dot_local[i]+1)
            M_dot[i]=M_dot[i+1]*multm_dot
    return M_dot
R = create_disc(5,3,10,2)

Mdot=M_dot(R, None, 100)   
print ("values for MDot",Mdot)

