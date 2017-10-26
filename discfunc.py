# -*- coding: utf-8 -*-

"""
discfunc.py

Created on Mon Oct  9 16:18:50 2017

Authors: Vysakh Salil, Norman Khan

Contains the main functions used for the accretion disk model 
Heavily based on the model proposed by P Arevalo P Uttley 2005
Can easily be imported by using:
    
    import discfunc as df
"""
#Imports
import numpy as np
import matplotlib.pyplot as plt

#================================================#
#====================FUNCTIONS===================#
#================================================#
def create_disc (N,const,Rmax,rstart):
    
    '''
    Creates a basic disc with constant ratio between radii
    Returns a python list array.
    
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


def M_dot(R,t,M_0_start):
    '''
    Calculates the mass accretion at a given radii between two annuli
    
    using the equation:
        M(r) = M_0(r) * ∏(1+m(r))
        
    where M(r) is the accretion rate at given radius
    M(r) is refered to as M_dot
    
    M_0(r) the LOCAL accretion rate at given radii
    M_0(r) is refered to as M_dot_local
    
    m_dot(r,t), the local small variation of mass at each radius
    can be modelled as a sinosoidal variation close to A*sin(2*pi*f*t)?
    where f is 1/(Viscous timescale)

    inputs:
        R         = Array of the radii
        t         = time
        M_0_start = Starting outer radius mass accretion rate

    '''
    
    #Creating Arrays to store values of M(r), M_0(r) and m(r)
    M_dot_local = [0]*len(R)
    M_dot_local[len(R)-1] = M_0_start
    #m_dot = [np.sin(2 * np.pi * (1/viscous_timescale(R)[i])* t) for i in range(len(R))]
    m_dot = [0.1*np.sin(2 * np.pi * (viscous_frequency(R)[i])* t) for i in range(len(R))]
    #print 'mdot', m_dot
    #multiplied mdot by 0.1 to more accurately reflect that mdot is << 1
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
    alpha=[np.random.uniform(0.01,0.1) for i in range(len(R))]
    return alpha


def viscous_frequency(R):
    '''
    Calculates the viscous frequency at a given radius
    based off the model by p.arevalo and p.uttley
    
    inputs:
        R         = Array of the radii
        M_0_start = Starting outer radius mass accretion rate
    '''
    f_visc=[]
    for i in range(len(R)):
        f_visc.append(((R[i])**(-3/2)) * ((H_R_)**2) * (alpha[i]/(2*np.pi)))
    return f_visc


def viscous_velocity(R):
    '''
    Calculates the viscous velocity at a given radius
    based off the model by p.arevalo and p.uttley
    
    inputs:
        R    = Array of the radii
        H_R_ = Starting outer radius mass accretion rate
    '''
    vel_visc=[]
    for i in range(len(R)):
        vel_visc.append(((R[i])**(-0.5))  *  ((H_R_)**2)  *  (alpha[i]))
    return vel_visc

def viscous_timescale (R):
    '''
    Calculates the viscous timescale at a given radius
    R/Viscous_velocity    or    R^2 / viscosity
    From eq 5.62 Accretion power in astrophysics.
    Time taken for the process to propogate from adjacent annulis
    '''
    time_visc=[]
    vel_visc=viscous_velocity(R)
    for i in range (len(R)):
        time_visc.append(R[i]/vel_visc[i])
    return time_visc

def emissivity (R):
    '''
    Calculates emissivity, which describes the total energy loss
    
    wiki:
        "Emissivity is defined as the ratio of the energy radiated from 
         a material's surface to that radiated from a blackbody"
    '''
    gamma=3
    em=[]
    for i in range(len(R)):
        em.append((R[i]**-gamma)*((R[0]/R[i])**0.5))
    return em
        

#================================================#
#====================CONSTANTS===================#
#================================================#


H_R_=0.01 # H/R (height of the disc over total radius)
M_0_start = 5


#================================================#
#=======================MAIN=====================#
#================================================#
'''Alphas are currently created once as a global variable 
due to random number generation'''
global alpha 

R = create_disc(5,3,10,2) 
alpha = calc_alpha(R)   



y=[]
T=[]

for t in np.arange(0,1e8,5000):
    #print 'time', t/10.0
    y.append(M_dot(R, t, M_0_start)[0])
    #y.append(sum(M_dot(R, t/10.0, M_0_start)))
    T.append(t)
    
    if t%500000==0:
        print t
plt.plot(T,y)

print '-------------------------------------'
print 'Radii:', R
print 'alphas:', alpha
print 'visc_freq:', viscous_frequency(R)
print 'visc_vel:', viscous_velocity(R)
print 'visc_timescale:', viscous_timescale(R)
print 'M_dot:', M_dot(R, 1, M_0_start)
