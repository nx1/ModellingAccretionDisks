# -*- coding: utf-8 -*-
import numpy as np


def M_dot(R,M_0_start):
    '''
    Calculates the mass accretion at a given radii between two annuli
    
    using the equation:
        M(r) = M_0(r) * ∏(1+m(r))
        
    where M(r) is the accretion rate at given radius
    M(r) is refered to as M_dot
    
    M_0(r) the LOCAL accretion rate at given radii
    M_0(r) is refered to as M_dot_local
    
    m(r) is a small stochastic variation in mass accretion rate << 1
    we currently make m(r) random between 0 and 0.1
         (WE DON'T KNOW HOW TO MAKE m(r) CURRENTLY)

    inputs:
        R         = Array of the radii
        M_0_start = Starting outer radius mass accretion rate

    '''
    
    #Creating Arrays to store values of M(r), M_0(r) and m(r)
    M_dot_local = [0]*len(R)
    M_dot_local[len(R)-1] = M_0_start
    m_dot = [np.random.uniform(0.0,0.1) for i in range(len(R))]
    M_dot = [0]*len(R)
    
    
    for i in range(len(R)):
        m_temp = 1 #Variable for storing 1 + mdot sum
        
        print('=================')
        print 'i =', i, '/', len(R)-1
        
        for j in range(i+1):
            print '-----------'
            print 'j = ', j, '/' , i
            print 'mdot', m_dot[-(1+j)]
            
            m_temp = m_temp * (1 + m_dot[-(1+j)])
            
            print 'm_temp = ', m_temp
            print '2+i = ', 2+i
            print 'len(R) = ', len(R)
        
    
        print('@@@@@@@@@@')
        M_dot[-(1+i)] = M_dot_local[-(1+i)] * m_temp
             
        if (2+i) < len(R)+1:
            M_dot_local[-(2+i)] = M_dot[-(1+i)]
        else:
            return M_dot


    
R = [1,5,9]
M_0_start = 100       
          
print (M_dot(R,M_0_start)) 


