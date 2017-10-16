import numpy as np

def blackbody(T):
    wavelength = 2900/T        
    return wavelength   #wavelength in micrometers.

#print(blackbody(500))



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


def M_dot(R, t):

    '''
    Calculates the mass accretion at a given radii between two annuli 
    
    inputs:
        R    = Number of Annuli
        t    = time (CURRENTLY UNUSED)

    '''
    
    print('INPUTS:', R)
    M_dot = []
    m_dot_local = [] 
    M_dot_local = []
    
    for i in range(len(R)):
        m_dot_local.append(np.random.uniform(0.0,0.1))
        M_dot_local.append(np.random.standard_cauchy())
        M_dot.append( M_dot_local[i] * (1 + m_dot_local[i]) )
        
        print('---------------', i, '---------------')
        print('Radius num:', i, 'radius value', R[i])
        
        print('M_dot', M_dot)
        print('M_dot_local', M_dot_local)
                        
M_dot(create_disc(5,3,10,2), None)
        
    