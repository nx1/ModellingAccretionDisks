# -*- coding: utf-8 -*-

"""
discfunc.py

Created on Mon Oct  9 16:18:50 2017

Authors: Vysakh Salil, Norman Khan

Contains the main functions used for the accretion disk model 
Heavily based on the model proposed by P Arevalo P Uttley 2005 
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time #remove once finished
from astroML.time_series.generate import generate_power_law
from astroML.fourier import PSD_continuous
from scipy import stats

seed = 5
#seed = np.random.random()
np.random.seed(seed)
#================================================#
#====================FUNCTIONS===================#
#================================================#

def create_disc (N,const,Rmin,Rmax):
    
    '''
    Creates a basic disc with constant ratio between radii
    Returns a python list array.
    
    inputs:
        N       = Number of Annuli
        const   = Constant ratio between consecutive radii
        Rmax    = Max Radius (CURRENTLY UNUSED)
        rstart  = Radii of innermost disc
    '''
    
    R = np.empty(N)
    for i in range(N):
        if i==0:
            R[i] = Rmin
        else:
            R[i] = R[i-1]*const
            
    return R
   

def calc_m_dot(R, timeSteps, Q):
    '''
    Calculates the local small mass accretion rate for all radius and time
    returns a 2 dimensional (len(R), timesteps) size ndarray
    
    Normalization is done by multiplying the output lightcurve by a factor
    X = sigma_var / standard deviation of m_dot
    
    inputs:
        R = array of radii
        timesteps = Number of equal-spaced time steps to generate 
    
    Uses modifed Timmer and Koenig method from AstroML with a Lorentzian
    distribution peaked at the local viscous frequency for the PSD. 
    '''
    F_var = 0.1     #Value of F_var used for normalisation eg 0.1 = 10%
    sigma_var = np.sqrt(F_var**2 / N)   #used to normalise over all annuli
    
    m_dot = np.empty((len(R), timeSteps))
    
    fVisc = viscous_frequency(R)
    
    for i in range(len(R)):
        m_dot[i] = generate_power_law(timeSteps, 1.0, Q[i], fVisc[i])
        X = sigma_var / np.std(m_dot[i])
        m_dot[i] = X * m_dot[i]
        
        percents = round(100.0 * i / float(len(R)), 1)
        if percents % 10.00==0:
            print percents, '% | calculating m_dot'
    return m_dot


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
        It will later be modelled as a power noise distribution as described
        by Timmer and Koenig (1995)

    inputs:
        R         = Array of the radii
        t         = time
        M_0_start = Starting outer radius mass accretion rate

    '''
    
    #Creating Arrays to store values of M(r), M_0(r) and m(r)
    M_dot_local = np.zeros(len(R))
    M_dot_local[len(R)-1] = M_0_start
    #m_dot = np.ones(len(R)) * 0.1*np.sin(2 * np.pi * (viscous_frequency(R))* t)
    #if you want to switch back to sinosoid, remove 2nd [t] term on m_dots
    
    #print 'mdot', m_dot
    
    #multiplied mdot by 0.1 to more accurately reflect that mdot is << 1
    M_dot = np.zeros(len(R))
    
    m_store = 1

    for i in range(len(R)-1,-1,-1):
        m_store=m_store*(m_dot[i][t]+1)
        if i==(len(R)-1):
            M_dot[i]=M_dot_local[i]*(m_dot[i][t]+1)
        else:
            M_dot[i]=M_dot[i+1]*m_store
    return M_dot


def dampen(M_dot, D):
    '''
    Dampens an input of M_dot at each radii by a factor
    exp(-D) based as done in Arevalo and Uttley 2005
    
    inputs:
        M_dot = list of mass accretion rates at each radius
        D = Damping coefficient 
    '''
    
    fourier = np.fft.fft(M_dot) * np.exp(-D)
    M_dot_new = np.fft.ifft(fourier)
    return M_dot_new
    
def calc_alpha (R):
    alpha = np.array([np.random.uniform(0.01,0.1) for i in range(len(R))])
    return alpha


def viscous_frequency(R):
    '''
    Calculates the viscous frequency at a given radius
    based off the model by p.arevalo and p.uttley
    
    inputs:
        R         = Array of the radii
        M_0_start = Starting outer radius mass accretion rate
    '''
    f_visc = R**(-1.5) * ((H_R_)**2.0) * alpha/(2.0*np.pi)
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
    vel_visc = 1/(2.0 * np.pi) * (R)**(-0.5)  *  (H_R_**2)  *  alpha
    return vel_visc


def viscous_timescale (R):
    '''
    Calculates the viscous timescale at a given radius
    R/Viscous_velocity    or    R^2 / viscosity
    From eq 5.62 Accretion power in astrophysics.
    Time taken for the process to propogate from adjacent annulus
    '''
    time_visc = R/viscous_velocity(R)
    return time_visc


def emissivity (R,gamma=3):
    '''
    Calculates emissivity, which describes the total energy loss
    
    wiki:
        "Emissivity is defined as the ratio of the energy radiated from 
         a material's surface to that radiated from a blackbody"
    '''
    em = (R**-gamma)*((R[0]/R)**0.5)
    return em


def T_eff (R):
    '''
    Caclulates the effective temperature of the disk at a given radius
    based off equation 5.43 in Accretion power in astrophysics
    (CONSTANTS OMMITED)
    '''
    T_eff = R**(-0.75)
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
    B_nu = 1 * nu**3 / ((np.e**(1 * nu /T_eff(R)))-1)
    return B_nu


def calc_area(R):
    '''
    Calculates the area of annuli from list of radii
    Returns len(R) - 1 list of areas
    '''
    A = np.array([np.pi * (R[i+1] ** 2 - R[i] ** 2) for i in range (len(R) - 1)])
    return A

def Lorentzian(T, Q):
    '''
    Calculates Lorentzian curves at each viscous frequency and sums them to
    create an psudo PSD. Useful for explaination.
    
    inputs:
        T = Array of time series, used to calculate frequency range
        Q = FWHM of the Lorentzians
    '''
    
    fVisc = viscous_frequency(R)
    f = np.arange(0, 1.2*max(fVisc), 1.2*max(fVisc)/len(T))
    S = np.empty(len(f))
    
    #j = 5   #Counter for plotting individual Lorentzians
    for i in range(len(fVisc)):
        S_store = (Q[i]/2.)**2 / ((f - fVisc[i])**2 + (Q[i]/2.)**2) 
        S_store_normalized = S_store * 1./max(S_store)
        
        plt.figure(4)
        plt.plot(f, S_store_normalized)
        #j = j + 1
        S = S + S_store_normalized
        #print 'S', S
    return S, f

#================================================#
#====================CONSTANTS===================#
#================================================#

H_R_= 1.0 # H/R (height of the disc over total radius) (10^-2 was suggested)
M_0_start = 1.0 #Starting M_0 at outermost radius
VERY_BIG = 1E50

#Disk constants
'''Arvelo and Uttley fix the first innermost radius at 6 Units
The number of annuli considered is also N = 1000
'''
N = 30      #Number of Radii
const = 1.1 #Constant of proportionality between neighbouring raddi radiuses.
Rmin = 6.0  #Minimum (starting) Radius
Rmax = 10.0

Q_factor = 0.025    #Value of FWHM of each Lorentzian
tMax_factor = 1.1   #Number of maximum viscous timescales to calculate to


#==================Variables=====================#
R = create_disc(N, const, Rmin, Rmax)
#alpha = 0.1*np.ones(len(R))
alpha = calc_alpha(R)           #Caclulates value of alpha at each radius
Q = Q_factor * viscous_frequency(R)  #FWHM of Lorentzians
tMax = int(tMax_factor * max(viscous_timescale(R)))
if tMax%2 != 0:       #PSD CALCULATION REQUIRES EVEN NUMBER OF TIMES
    tMax = tMax + 1
    
#================================================#
#=======================MAIN=====================#
#================================================#
time0 = time()

print '--------- Calculating m_dot (small) ---------'
print 'radii:', len(R), '| tMax:', tMax, '| tMax factor:', tMax_factor
m_dot = calc_m_dot(R,tMax, Q)
print 'DONE in: ', time() - time0
print '---------------------------------------------'   
print ''                    

print '============== DISK PARAMETERS =============='
np.set_printoptions(precision = 2, linewidth = 100)
print 'Number of radii:', N
print 'Radii:', R
print 'alphas:', alpha
print 'visc_timescale:', viscous_timescale(R)
np.set_printoptions(precision = 4, linewidth = 100)
print 'visc_freq: ', viscous_frequency(R)
print 'visc_vel:', viscous_velocity(R)
print 'M_dot: ', M_dot(R, 1, M_0_start)
print '============================================='
print ''
print '############# CALCULATING M_DOT #############'

############-Count Rate vs Time-############

y=np.empty(tMax)
T=np.empty(tMax)
for t in np.arange(0,tMax,1):
    #y[t] = dampen(M_dot(R, t, M_0_start), 0.5)[0]    #Damped
    y[t] = M_dot(R, t, M_0_start)[0]
    T[t] = t
    
    percents = round(100.0 * t / float(tMax), 4)
    if percents % 10.00==0:
       print percents, '%', '|Calculating M_dot| t =', t, '/', tMax
    
fig1 = plt.figure(1, figsize=(7, 4))  

visctime = viscous_timescale(R)

for i in visctime:
    plt.axvline(x = i, linewidth = 0.5)

plt.title('Light Curve for disk of %d radii' %N)  
plt.xlabel('time')
plt.ylabel('Mass accretion at R[0]')        
plt.plot(T,y , linewidth=0.25)
print '#############################################'
print ''


############-RMS vs Average Count Rate-############

a = np.array_split(T, len(T)/(tMax/10))
count_bin = np.array_split(y, len(y)/(tMax/10))


b_avg = np.empty(len(a))
b_rms = np.empty(len(a))

for i in range(len(a)):
    b_avg[i] = np.average(count_bin[i])
    #print 'bavg:', b_avg
    b_rms[i] = np.sqrt(np.average(count_bin[i] ** 2))
    #print 'brms:', b_rms


fig2 = plt.figure(2, figsize=(7, 7)) 
plt.xlabel('average count rate')
plt.ylabel('rms') 
fit = np.polyfit(b_avg,b_rms,1)
fit_fn = np.poly1d(fit) 
plt.plot(b_avg,fit_fn(b_avg), color='r')

plt.scatter(b_avg,b_rms, marker='x')

slope, intercept, r_value, p_value, std_err = stats.linregress(b_avg,b_rms)
print '========REGRESSION STATS FOR RMS========'
print'slope:', slope
print 'intercept:', intercept
print 'r_value:', r_value
print 'p_value:', p_value
print 'std_err:', std_err
print '========================================'



############ PSD ############

freq, PSD = PSD_continuous(T,y)

fig3 = plt.figure(3, figsize=(7, 4))
plt.title('PSD')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.loglog(freq,PSD, linewidth=0.5, color='black')

viscfreq = viscous_frequency(R)

for i in viscfreq:
    plt.axvline(x = i, linewidth = 0.5)
    
############################







'''
##########PSD from Lorentzian combination###########

S_f, Freq = Lorentzian(T, Q)

plt.figure(4)
plt.title('PSD Lorentzian combination')
plt.plot(Freq, S_f)
for i in viscfreq:
    plt.axvline(x = i, linewidth = 0.5, color = 'r')





#------------------------------------
#Flux vs radius
B = B_nu(R, 1)
B = np.delete(B,-1)
A = calc_area(R)

flux = A*B
R_new = np.delete(R,-1)

plt.figure(4)
plt.title('Radius vs flux for frequency = 1 in loglog')
plt.xlabel('Radius')
plt.ylabel('Flux')
plt.loglog(R_new,flux)

fourier = np.fft.fft(y)
freq = 1/T



#------------------------------------
#Total Flux vs frequency

Bnew=[]
flux_total=[]
F=np.arange(1,30,0.01)
for f in F:
    flux =[]
    for i in range (len(R)-1):
        Bnew=B_nu(R, f)
        flux.append(A[i] * Bnew[i])
    flux_total.append(sum (flux) )

plt.figure(5)
F_array = np.asarray(F)
plt.title('Frequency vs total flux for varying frequencies in loglog')
plt.xlabel('Frequency')
plt.ylabel('Total Flux')
plt.loglog(F_array,flux_total)
#------------------------------------
#Emissivity Flux vs radius
em=emissivity(R)
em_flux = []
R_new = []


for i in range (len(R)-1):
    em_flux.append(A[i]*em[i])
    R_new.append(R[i])

plt.figure(6)
plt.title('Radius vs em flux for frequency = 1 in loglog')
plt.xlabel('Radius')
plt.ylabel('em Flux')
plt.loglog(R_new,em_flux)

'''

plt.show()
time1 = time()
print 'Time taken', time1-time0, '| seed:', seed
