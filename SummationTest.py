# -*- coding: utf-8 -*-
import numpy as np



R = [1,5,9,5,5]

M_0_start = 100

M_dot_local = [0]*len(R)
M_dot_local[len(R)-1] = M_0_start
            
m_dot = [np.random.random() for i in range(len(R))]  

M_dot = [0]*len(R)

    
for i in range(len(R)):
    m_temp = 0 #Variable for storing 1 + mdot sum

    for j in range(i+1):
        print(m_dot[-(1+j)])
        m_temp = m_temp * (1 + m_dot[-(1+j)])
        
        
    print(M_dot_local[-(1+i)])
    print(i)
    
          
          
'''
i = 0 ----> * 1
i = 1 ----> * 2
'''


