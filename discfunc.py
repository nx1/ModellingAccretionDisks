# -*- coding: utf-8 -*-

"""
discfunc.py

Created on Mon Oct  9 16:18:50 2017

Authors: Vysakh Salil, Norman Khan

This program contains a model for a fluctuating accretion disk based on 
the model put forward by P. Arevalo and P. Uttley in 2005
DOI: 	10.1111/j.1365-2966.2006.09989.x
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time #remove once finished
from astroML.time_series.generate import generate_power_law
from astroML.fourier import PSD_continuous
from scipy import stats
from scipy import signal
#seed = 5
seed = int(100*np.random.random())
np.random.seed(seed)

#================================================#
#====================FUNCTIONS===================#
#================================================#

def create_disc (N,Rmin,Rmax):
    '''
    Creates a basic disc with constant ratio between radii
    Returns a python list array.
    
    inputs:
        N       = Number of Annuli
        Rmin    = Radii of innermost disc
        Rmax    = Radii of outermost disc
    '''
    R = np.empty(N)
    const = np.power(Rmax/Rmin, 1./(N-1))
    
    for i in range(N):
        R[i] = const**i * Rmin   
    return R
   

def calc_m_dot(R, timeSteps, Q, sinusoid):
    '''
    Calculates the local small mass accretion rate for all radius and time
    returns a 2 dimensional (len(R), timesteps) size ndarray
    
    Normalization is done by multiplying the output lightcurve by a factor
    X = sigma_var / standard deviation of m_dot
    
    inputs:
        R = array of radii
        timesteps = Number of equal-spaced time steps to generate 
        Q = Quality factor Lorentzian peak freq / FWHM
        
    Uses modifed Timmer and Koenig method from AstroML with a Lorentzian
    distribution peaked at the local viscous frequency for the PSD. 
    
    Q = Q_factor * viscous_frequency(R) Where Q factor is between 0.5 to 10
    Q = 10 produces a narrower PSD
    '''
    print '--------- Calculating m_dot (small) ---------' 
    m_dot = np.empty((len(R), timeSteps))#Used to store rxt array of m_dot
    fVisc = viscous_frequency(R)
    
    if not sinusoid:
        F_var = 0.1     #Value of F_var used for normalisation eg 0.1 = 10%
        sigma_var = np.sqrt(F_var**2 / N)   #used to normalise over all annuli

        for i in range(len(R)): #Calculates m_dot for all radius and time
            print 'Calculating m_dot | R =', i+1, '/', len(R)
            m_dot[i] = generate_power_law(timeSteps, 1.0, Q, fVisc[i])
            X = sigma_var / np.std(m_dot[i])    #Normalisation
            m_dot[i] = X * m_dot[i]             #Normalisation
            
    else:
        for r in range(len(R)): #Calculates m_dot for all radius and time
            for t in range(timeSteps):
                m_dot[r][t] = 0.1*np.sin(2.0 * np.pi * (fVisc[r]) * t)
    print '---------------------DONE--------------------' 
        
    return m_dot


def M_dot(R,t,Mdot0):
    '''
    Calculates the mass accretion at a given radii between two annuli
    
    using the equation:
        M(r) = Mdot0 * ∏(1+m(r))
        
    where M(r) is the accretion rate at given radius
    M(r) is refered to as Mdot
    
    Mdot0 is the outermost radius accretion rate 
    
    m_dot(r,t), the local small variation of mass at each radius

    inputs:
        R         = Array of the radii
        t         = time
        Mdot0     = Outermost radius accretion rate
    
    '''
    #
    
    Mdot = np.zeros(len(R))
    
    for i in range(len(R)-1,-1,-1): #Runs through all annuli from out to in
        #print 'Radius # ', i
        if i==(len(R)-1):
            Mdot[i]=Mdot0*(1.0 + m_dot[i][t])
        else:
            Mdot[i]=Mdot[i+1]*(1.0 + m_dot[i][t])
    return Mdot

    
def calc_alpha (R):
    '''
    Calculates alpha for all radii
    '''
    #alpha = np.array([np.random.uniform(0.01,0.1) for i in range(len(R))])
    alpha = 0.3
    return alpha


def viscous_frequency(R):
    '''
    Calculates the viscous frequency at a given radius
    '''
    f_visc = viscous_velocity(R)/R
    return f_visc


def viscous_velocity(R):
    '''
    Calculates the viscous velocity at a given radius
    Also sometimes called the radial drift velocity.
    '''
    vel_visc = 1/(2.0 * np.pi) * (R)**(-0.5)  *  (H_R_**2)  *  alpha
    return vel_visc


def viscous_timescale (R):
    '''
    Calculates the viscous timescale at a given radius
    From eq 5.62 Accretion power in astrophysics.
    Time taken for the process to propogate from annulus to centre
    '''
    time_visc = R/viscous_velocity(R)
    return time_visc

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



def emissivity (R,gamma):
    '''
    Calculates emissivity, which is used to find the flux of the disc
    as the mass accretion rate at a given radius is proportional to its
    emmissivity at that radius.
    
    gamma is used to probe different parts of the x-ray spectra, eg soft = 3
    hard = 5
    '''
    em = (R**-gamma)*((R[0]/R)**0.5)
    return em


def T_eff (R):
    '''
    Caclulates the effective temperature of the disk at a given radius
    based off equation 5.43 in Accretion power in astrophysics
    (CONSTANTS OMMITED)
    '''
    T_eff = np.empty((len(R),tMax))
    for i in range (len(R)):
        for t in range(tMax):
            T_eff[i][t]= ((M[i][t]/R[i]**3)*(1-(R[0]/R[i])**0.5))**0.25
        
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
    A = np.array([np.pi * (R[i+1] ** 2. - R[i] ** 2.) for i in range (len(R) - 1)])
    return A

def calc_flux(R, nu):
    '''
    Calculates the flux via blackbody spectrum at all radius and all time
    returns rxt array
    inputs:
        R = list of Radii
        nu = frequency
    '''
    A = calc_area(R)
    B = B_nu(R, nu)
    Flux = np.empty((len(R)-1,tMax))
    for i in range(len(R)-1):
        for t in range(tMax):        
            Flux[i][t] = A[i] * B[i][t]
    return Flux
    
def average_arr(arr):
    '''
    Calcutes the average value between each item in an array arr
    return length len(arr) - 1 array of new values
    '''
    arr1 = np.empty(len(arr)-1)
    for i in range(len(arr)-1):
        arr1[i] = (arr[i] + arr[i+1]) / 2.
    return arr1

def shift_M_dot(M_dot):
    '''
    Calculates the shift in the mass accretion rate by the propagation time from
    the outermost annulus to the present annulus
    '''
    ViscTime = viscous_timescale(R)
    ViscTimeMax = int(max(ViscTime))
    
    deltaT = np.empty(N)        #To insert
    deltaT_append = np.empty(N) #To append
    
    
    for i in range(N):                
        deltaT[i] = int(ViscTimeMax - ViscTime[i])
        deltaT_append[i] = deltaT[0] - deltaT[i]
        
    print 'deltaT:', deltaT
    print 'append:', deltaT_append
        
    M_shifted = np.empty((N, len(M_dot[0]) + deltaT[0]))
    print 'M_dot length:', 
    print 'M_shifted length:', len(M_shifted[0])
    print 'MDOT'
    print M_dot
    for r in range(N):
        print 'MDOT', r
        print M_dot[r]
        
        M_inserted = np.insert(M_dot[r], 0, np.zeros(deltaT[r]))     
        M_shifted[r] = np.append(M_inserted, np.zeros(deltaT_append[r]))

    return M_shifted

def filter_factor(R, rf, gamma):
    '''
    Calculates the Filter factor for Normalisation of the PSD
    See appendix arevalo, Uttley
    Inputs:
        R = as array
        Gamma = as int
        rf = final radius for filter factor
    '''
    em = emissivity(R,gamma)
    area = calc_area(R)
    em_avg = average_arr(em_soft)
    
    
    flux_array  = area * em_soft_avg
    
    flux_tot = np.sum(flux_array)
    flux_rf = np.sum(flux_array[:rf])
    
    filter_factor = (flux_rf/flux_tot)**2.
    return filter_factor

#================================================#
#====================CONSTANTS===================#
#================================================#

H_R_= 1.0 # H/R (height of the disc over total radius) (10^-2 was suggested)
M_0_start = 1.0 #Starting M_0 at outermost radius

#Disk constants
'''Arvelo and Uttley fix the first innermost radius at 6 Units
The number of annuli considered is also N = 1000
'''
N = 5      #Number of Radii
Rmin = 1.0  #Minimum (starting) Radius
Rmax = 10.0

Q = 0.5  #Q is defined as the ratio of lorenzian peak freqency to FWHM
tMax_factor = 10.1   #Number of maximum viscous timescales to calculate to



#==================Variables=====================#
R = create_disc(N, Rmin, Rmax)
#alpha = 0.1*np.ones(len(R))
alpha = calc_alpha(R)           #Caclulates value of alpha at each radius

ViscMax = max(viscous_timescale(R))
tMax = int(tMax_factor * ViscMax)
if tMax%2 != 0:       #PSD CALCULATION REQUIRES EVEN NUMBER OF TIMES
    tMax = tMax + 1
    
sinusoid = False    #Creates m_dot as either sinusoids or timmer and koenig  
          
m_dot = calc_m_dot(R, tMax, Q, sinusoid)    
    
    
    
    
#================================================#
#=======================MAIN=====================#
#================================================#
time0 = time()
print ''
print 'radii:', len(R), '| tMax:', tMax, '| tMax factor:', tMax_factor, '| ViscMax', ViscMax
print '---------------------------------------------'   
print ''                    
print '============== DISK PARAMETERS =============='
np.set_printoptions(precision = 2, linewidth = 100)
print 'Number of radii:', N
print 'alphas:', alpha
print 'Radii:         ', R
print 'visc_timescale:', viscous_timescale(R)
np.set_printoptions(precision = 4, linewidth = 100)
print 'visc_freq:     ', viscous_frequency(R)
print 'visc_vel:      ', viscous_velocity(R)
print 'M_dot:         ', M_dot(R, 1, M_0_start)
print '============================================='
print ''
print '############# CALCULATING M_DOT #############'




############-Count Rate vs Time-############


M=np.empty((tMax,N))
T=np.arange(tMax)

for t in np.arange(0,tMax,1):   
    
    #M[t] = M_dot(R, t, M_0_start)
    M[t] = dampen(M_dot(R, t, M_0_start), 0.5) #damped
    percents = round(100.0 * t / float(tMax), 4)
    if percents % 10.00==0:
       print percents, '%', '|Calculating M_dot| t =', t, '/', tMax
  
M = np.transpose(M)
fig1 = plt.figure(1, figsize=(7, 4))  

visctime = viscous_timescale(R)

for i in visctime:  #Vertical lines at each viscous timescale
    plt.axvline(x = i, linewidth = 0.25, color='green')

plt.title('Mass accretion rate(s) for disk of %d radii' %N)  
plt.xlabel('time')
plt.ylabel('Mass accretion at R[0]')     
for i in range(N): plt.plot(T,M[i], linewidth=0.25, label='r = %d'%i)
#plt.legend()
print '#############################################'
print ''


############-RMS vs Average Count Rate-############

a = np.array_split(T, len(T)/(tMax/10))
count_bin = np.array_split(M[0], len(M[0])/(tMax/10))

b_avg = np.empty(len(a))
b_rms = np.empty(len(a))

for i in range(len(a)):
    b_avg[i] = np.average(count_bin[i])
    #print 'bavg:', b_avg
    b_rms[i] = np.sqrt(np.average(count_bin[i] ** 2))
    #print 'brms:', b_rms


fig2 = plt.figure(2, figsize=(7, 7)) 
plt.xlabel('Average count rate')
plt.ylabel('rms') 

plt.plot(np.unique(b_avg), np.poly1d(np.polyfit(b_avg, b_rms, 1))(np.unique(b_avg)), color='r')

plt.scatter(b_avg,b_rms, marker='x')

slope, intercept, r_value, p_value, std_err = stats.linregress(b_avg,b_rms)
print '========REGRESSION STATS FOR RMS========'
print'slope:', slope
print 'intercept:', intercept
print 'r_value:', r_value
print 'p_value:', p_value
print 'std_err:', std_err
print '========================================'


############# Light Curves From Emissivity #############

em_soft = emissivity(R,3)  #Calculates emissivity at every radius (soft)
em_hard = emissivity(R,5)  #Calculates emissivity at every radius (hard)

em_soft_avg = average_arr(em_soft)
em_hard_avg = average_arr(em_hard)

area = calc_area(R)

flux_soft = area * em_soft_avg
flux_hard = area * em_hard_avg

M_shifted = shift_M_dot(M)

M_scaled_soft = np.empty((N-1, len(M[0])))   #Used to store the new scaled lightcurves
M_scaled_hard = np.empty((N-1, len(M[0])))   #Used to store the new scaled lightcurves



for i in range(N-1):
    M_scaled_soft[i] = flux_soft[i]/max(flux_soft) * M[i]    #normalised to 1
    M_scaled_hard[i] = flux_hard[i]/max(flux_hard) * M[i]    #normalised to 1


M_total_soft = np.sum(M_scaled_soft, axis=0) / np.max(np.sum(M_scaled_soft, axis=0))
M_total_hard = np.sum(M_scaled_hard, axis=0) / np.max(np.sum(M_scaled_hard, axis=0))


fig4 = plt.figure(4, figsize=(7, 4))
plt.title('Lightcurve from emissivity for soft/hard state')
plt.xlabel('Time')
plt.ylabel('Count')

#T = np.arange(0,len(M_total_soft),1)

#sum of all radii is total contribution
plt.plot(T,M_total_soft, label='soft', color='black', linewidth = 0.5) 
plt.plot(T,M_total_hard, label='hard', color='red', linewidth = 0.5)
for i in visctime:  #Vertical lines at each viscous timescale
    plt.axvline(x = i, linewidth = 1.0, color='green', linestyle='--') 
plt.legend()


cor=np.correlate(M_total_soft,M_total_hard, mode='full')

fig6 = plt.figure(6, figsize=(7, 4))

plt.plot(np.arange(0,len(cor),1),cor)

#plt.legend()

########################################################




############ PSD ############

freq, PSD = PSD_continuous(T,M_total_soft)


fig3 = plt.figure(3, figsize=(7, 7))
plt.title('PSD at for total light curve soft')
plt.xlabel('Frequency')
plt.ylabel('f*P(f)')
#plt.loglog(freq,PSD*freq, linewidth=0.25, color='black')


ff=filter_factor(R, 1, 3)

plt.loglog(freq, PSD*freq, linewidth=0.25, color='black') 

plt.loglog(np.unique(freq), np.poly1d(np.polyfit(freq, PSD*freq, 3))(np.unique(freq)))
viscfreq = viscous_frequency(R)

for i in viscfreq:
    plt.axvline(x = i, linewidth = 1.0, linestyle='--')



'''
############# Light Curves From Blackbody #############

fig5 = plt.figure(5, figsize=(7, 4))
plt.xlabel('Radius')
plt.ylabel('flux')
for i in np.arange(0, 10. , 1.):
    flux2 = calc_flux(R,i)
    plt.plot(R[:-1],np.transpose(flux2)[0], label=i)
plt.legend()
'''


plt.show()
time1 = time()
print 'Time taken', time1-time0, '| seed:', seed
    
'''
############ PSD for m_dot ############
radius = 0
freq1, PSD1 = PSD_continuous(T,m_dot[radius])

fig8 = plt.figure(8, figsize=(7, 7))
plt.title('PSD for m_dot[%s]' %radius)
plt.xlabel('Frequency')
plt.ylabel('f*P(f)')
plt.loglog(freq1,PSD1*freq1, linewidth=0.25, color='black')
viscfreq = viscous_frequency(R)

for i in viscfreq:
    if i == viscfreq[radius]:
        plt.axvline(x = i, linewidth = 1.0, color='red', linestyle='--')
    else:
        plt.axvline(x = i, linewidth = 1.0, linestyle='--')
'''