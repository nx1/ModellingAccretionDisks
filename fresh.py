# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from time import time #remove once finished
from astroML.time_series.generate import generate_power_law
from astroML.fourier import PSD_continuous
from scipy import stats
from scipy.stats import cauchy




T = np.arange(0,1000,0.1)

def Lorentzian(T, Q):
    f = 1./T
    fVisc = []
    #for i in range(len(fVisc)):
    S = (1./np.pi * (Q/2.)) * ( (Q/2.)**2 / ((f - 5)**2 + (Q/2.)**2))
        
    return S

S = Lorentzian(T, 1)
 
plt.figure(1)

plt.plot(1./T , S)


plt.figure(2) 

plt.plot(T,S)