# -*- coding: utf-8 -*-
import numpy as np

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
            
    return np.asarray(R)    #Converts python list to numpy array


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
        for j in range(i+1):
            m_temp = m_temp * (1 + m_dot[-(1+j)])

        M_dot[-(1+i)] = M_dot_local[-(1+i)] * m_temp
             
        if (2+i) < len(R)+1:
            M_dot_local[-(2+i)] = M_dot[-(1+i)]
        else:
            return M_dot
    
    
def calc_alpha (R):
    alpha=[]
    for i in range(len(R)):
        alpha.append(np.random.random())
    return alpha


def viscous_frequency(R):
    '''
    Calculates the viscous frequency at a given radius
    based off the model by p.arevalo and p.uttley
    
    '''
    f_visc=[]
    for i in range(len(R)):
        f_visc.append(((R[i])**(-3/2)) * ((H_R_)**2) * (alpha[i]/(2*np.pi)))
    return f_visc


def viscous_velocity(R):
    '''
    Calculates the viscous velocity at a given radius
    based off the model by p.arevalo and p.uttley
    
    '''
    vel_visc=[]
    for i in range(len(R)):
        vel_visc.append(((R[i])**(-0.5))  *  ((H_R_)**2)  *  (alpha[i]))
    return vel_visc

def viscous_timescale (R):
    '''
    Calculates the viscous timescale at a given radius
    R/Viscous_velocity
    
    '''
    time_visc=[]
    vel_visc=viscous_velocity(R)
    for i in range (len(R)):
        time_visc.append(R[i]/vel_visc[i])
    return time_visc
        

#================================================#


H_R_=0.01 # H/R (height of the disc over total radius)
M_0_start = 100


#================================================#
R = create_disc(5,3,10,2) 
alpha = calc_alpha(R)

print('alphas:', alpha)
print('visc_freq', viscous_frequency(R)) 
print('visc_vel', viscous_velocity(R))
print('visc_time', viscous_timescale(R))

print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                
print M_dot(R, M_0_start)    

print('------')







    