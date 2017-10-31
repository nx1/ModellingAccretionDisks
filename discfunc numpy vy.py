# -*- coding: utf-8 -*-

"""
discfunc.py

Created on Mon Oct  9 16:18:50 2017

Authors: Vysakh Salil, Norman Khan

Contains the main functions used for the accretion disk model 
Heavily based on the model proposed by P Arevalo P Uttley 2005

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
    m_dot = [0.1*np.sin(2 * np.pi * (viscous_frequency(R)[i])* t) 
            for i in range(len(R))]
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
    f_visc=np.array([])
    #f_visc=[]
    for i in range(len(R)):
        np.append(f_visc,(((R[i])**(-3/2)) * ((H_R_)**2) * (alpha[i]/(2*np.pi))))
        #f_visc.append(((R[i])**(-3/2)) * ((H_R_)**2) * (alpha[i]/(2*np.pi)))
    return f_visc


def viscous_velocity(R):
    '''
    Also sometimes called the radial drift velocity.
    Calculates the viscous velocity at a given radius
    based off the model by p.arevalo and p.uttley
    
    inputs:
        R    = Array of the radii
        H_R_ = Starting outer radius mass accretion rate
    '''
    vel_visc= np.array([])
    #vel_visc=[]
    for i in range(len(R)):
        np.append(vel_visc,(((R[i])**(-0.5))  *  ((H_R_)**2)  *  (alpha[i])))
        #vel_visc.append(((R[i])**(-0.5))  *  ((H_R_)**2)  *  (alpha[i]))
    return vel_visc

def viscous_timescale (R):
    '''
    Calculates the viscous timescale at a given radius
    R/Viscous_velocity    or    R^2 / viscosity
    From eq 5.62 Accretion power in astrophysics.
    Time taken for the process to propogate from adjacent annulus
    '''
    time_visc= np.array([])
    #time_visc=[]
    vel_visc=viscous_velocity(R)
    for i in range (len(R)):
        np.append(time_visc,(R[i]/vel_visc[i]))
        #time_visc.append(R[i]/vel_visc[i])
    return time_visc

def emissivity (R):
    '''
    Calculates emissivity, which describes the total energy loss
    
    wiki:
        "Emissivity is defined as the ratio of the energy radiated from 
         a material's surface to that radiated from a blackbody"
    '''
    gamma=3
    em=np.array([])
    for i in range(len(R)):
        np.append(em,((R[i]**-gamma)*((R[0]/R[i])**0.5)))
        #em.append((R[i]**-gamma)*((R[0]/R[i])**0.5))
    return em

def T_eff (R):
    '''
    Caclulates the effective temperature of the disk at a given radius
    based off equation 5.43 in Accretion power in astrophysics
    (CONSTANTS OMMITED)
    '''
    np.array([R[i]**(-0.75) for i in range(len(R))])
    #T_eff = [R[i]**(-0.75) for i in range(len(R))]
    return T_eff

def B_nu(R, nu):
    '''
    Calculates the blackbody spectral irradiance based off eq 5.44.2
    Accretion power in astrophysics (CONSTANTS OMMITED)
    
    Inputs:
        R = list of radii
        nu = frequency   
    
    Note: B_nu is large at small radii however it is a measure of density.
    '''
    B_nu=[]
    for i in range(len(R)):
        B_nu.append( 1 * nu**3 / ((np.e**(1 * nu /T_eff(R)[i]))-1) )
    return B_nu

def calc_area(R):
    '''
    Calculates the area of annuli from list of radii
    Returns len(R) - 1 list of areas
    '''
    A = [np.pi * (R[i+1] ** 2 - R[i] ** 2) for i in range (len(R) - 1)]
    return A
#================================================#
#====================CONSTANTS===================#
#================================================#


H_R_=0.01 # H/R (height of the disc over total radius)
M_0_start = 1
VERY_BIG = 1E50


#================================================#
#=======================MAIN=====================#
#================================================#

'''Arvelo and Uttley fix the first innermost radius at 6 Units
The number of annuli considered is also N = 1000
'''
#N, ratio, rmax, rmin
R = create_disc(30,1.1,10,6)


'''Alphas are currently created once as a global variable 
due to random number generation'''
global alpha 
alpha = calc_alpha(R)   



y=[]
T=[]

tMax = 1e9
'''
for t in np.arange(0,tMax,50000):
    #print 'time', t/10.0
    y.append(M_dot(R, t, M_0_start)[0])
    #y.append(sum(M_dot(R, t/10.0, M_0_start)))
    T.append(t)
    
    if t%(tMax/10)==0:
        print (t/tMax)*100 , '%'
   
    
plt.figure(1)    
plt.xlabel('time')
plt.ylabel('Mass accretion at R[0]')        
plt.plot(T,y)

#------------------------------------
#Attempted PSD (completely wrong)
plt.figure(2)  
plt.xlabel('frequency')
plt.ylabel('Fourier transform of something * f')  
y2 = y*np.fft.fft(y)      #fast fourier transform * freq
f = 1 / np.asarray(T)   #1/t is basically frequency
plt.semilogx(f,y2)

#------------------------------------
'''
#------------------------------------
#Flux vs radius
B=B_nu(R, 1)
A=calc_area(R)
flux = []
R_new=[]


for i in range (len(R)-1):
    flux.append(A[i]*B[i])
    R_new.append(R[i])

plt.figure(3)
plt.title('Radius vs flux for frequency = 1 in loglog')
plt.xlabel('Radius')
plt.ylabel('Flux')
plt.loglog(R_new,flux)


Bnew=[]
flux_total=[]
F=np.arange(1,20,0.01)
for f in F:
    flux =[]
    for i in range (len(R)-1):
        Bnew=B_nu(R, f)
        flux.append(A[i] * Bnew[i])
    flux_total.append(sum (flux) )

plt.figure(4)

F_array = np.asarray(F)
plt.title('Frequency vs total flux for varying frequencies in loglog')
plt.xlabel('Frequency')
plt.ylabel('Total Flux')
plt.loglog(F_array,flux_total)
#------------------------------------




print '-------------------------------------'
print 'Radii:', R
print 'alphas:', alpha
print 'visc_timescale:'
print ['%E' % tf for tf in viscous_timescale(R)]
print 'visc_freq: '
print ['%E' % tf for tf in viscous_frequency(R)]
print 'visc_vel:'
print [' %E' % tf for tf in viscous_velocity(R)]
print 'M_dot: '
print ['%E' % tf for tf in M_dot(R, 1, M_0_start)]
print M_dot(R, 1, M_0_start)
print '-------------------------------------'
