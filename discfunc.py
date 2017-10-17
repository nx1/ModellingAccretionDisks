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
          
        
    
    return R_new, Theta


def M_dot(R, t, M_0_Start):

    '''
    Calculates the mass accretion at a given radii between two annuli 
    
    inputs:
        R    = Array of the radii
        t    = time (CURRENTLY UNUSED)

    '''
    
    print('INPUTS:', R)

    M_dot = np.empty(len(R))
    m_dot_local = np.empty(len(R))
    M_dot_local = np.empty(len(R))
    M_dot_local[len(R)-1]=M_0_Start
    print m_dot_local
    

    M_dot = [None] * len(R)
    m_dot_local = [None] * len(R)
    M_dot_local = [None] * len(R)
    M_dot_local[len(R)-1] = M_0_Start
                
    print(M_dot_local)

    for i in range(len(R)):
        np.append(m_dot_local,np.random.uniform(0.0,0.1))
        #m_dot_local is <<1
        print ("mdot",m_dot_local)
        """
        M_dot_local.append(np.random.standard_cauchy())
        M_dot.append( M_dot_local[i] * (1 + m_dot_local[i]) )
        
        print('---------------', i, '---------------')
        print('Radius num:', i, 'radius value', R[i])
        
        print('M_dot_local', M_dot_local)
        print('M_dot', M_dot)
        
    return M_dot
    
  """  
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



#================================================#
R = create_disc(5,3,10,2) 
alpha = calc_alpha(R)

print('alphas:', alpha)
print('visc_freq', viscous_frequency(R)) 
print('visc_vel', viscous_velocity(R))
print('visc_time', viscous_timescale(R))

print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                
M_dot(R, None, 100)    

print('------')







    