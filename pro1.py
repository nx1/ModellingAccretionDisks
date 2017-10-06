
from math import *
import pylab
M=1.989*(10**30) #mass of binary stars (kg)
G=6.67408*(10**-11) #Gravitational constant (m^3/kg^-1 x s^-2)
R=149.6*(10**5) #Radaius of circle (m)
r=30*R
om=(G*M/R**3)**0.5 #omega
t=0 #
t1=0# intial times
t2=pi*(r/R)**1.5 #time period



def runkut1(n, t, y, h):
    "Advances the solution of diff eqn defined by derivs from t to t+h" 
    y0=y[:]
    k1=derivs1(n, t, y)
    for i in range(1,n+1): 
        y[i]=y0[i]+0.5*h*k1[i]
        k2=derivs1(n, t+0.5*h, y)
    for i in range(1,n+1):
        y[i]=y0[i]+h*(0.2071067811*k1[i]+0.2928932188*k2[i])
        k3=derivs1(n, t+0.5*h, y)
    for i in range(1,n+1):
        y[i]=y0[i]-h*(0.7071067811*k2[i]-1.7071067811*k3[i])
        k4=derivs1(n, t+h, y)
    for i in range(1,n+1):
        a=k1[i]+0.5857864376*k2[i]+3.4142135623*k3[i]+k4[i]
        y[i]=y0[i]+0.16666666667*h*a
    t+=h
    return (t,y)
    
def derivs1(n, t, y):
    "this function calculates y' from t and y"
    dy=[0 for i in range(0,n+1)]
    dy[1]=y[2]
    dy[2]=-2*G*M*y[1]/r**3#two ODEs
    return dy

def runkut2(n, t, x, h):
    "Advances the solution of diff eqn defined by derivs from t to t+h" 
    x0=x[:]
    k1=derivs2(n, t, x)
    for i in range(1,n+1): 
        x[i]=x0[i]+0.5*h*k1[i]
        k2=derivs2(n, t+0.5*h, x)
    for i in range(1,n+1):
        x[i]=x0[i]+h*(0.2071067811*k1[i]+0.2928932188*k2[i])
        k3=derivs2(n, t+0.5*h, x)
    for i in range(1,n+1):
        x[i]=x0[i]-h*(0.7071067811*k2[i]-1.7071067811*k3[i])
        k4=derivs2(n, t+h, x)
    for i in range(1,n+1):
        a=k1[i]+0.5857864376*k2[i]+3.4142135623*k3[i]+k4[i]
        x[i]=x0[i]+0.16666666667*h*a
    t+=h
    return (t,x)
    
def derivs2(n, t, x): 
    "this function calculates x' from t and x"
    dx=[0 for i in range(0,n+1)]
    dx[1]=x[2]
    dx[2]=-2*G*M*x[1]/r**3#two ODEs
    return dx

N=1000 #number of times runkut1 and runkut2 will be called
z=[0 for j in range(0,N)]
l=[0 for j in range(0,N)]


y=[0, 1.0, 0.0]; x=[0, 0.0, 1.0]# Sets Boundary Conditions

for j in range(0,N):# run loop to updated values and equate to arrays
    (t,y) = runkut1(2, t, y, 3750.0/N) #calls runkut1 with present stepsize 3750/N
    (t1,x) = runkut2(2, t1, x, 3750.0/N)#calls runkut2 with present stepsize 3750/N
    print t1, t, y[1], y[2], x[1], x[2] #used to check all the values 
    z[j]=y[1]# Equating the new updated values to arrays
    l[j]=x[1]#

pylab.plot(l,z)#plot motion of planet
pylab.xlabel('x coordinates (units R)')
pylab.ylabel('y coordinates (units R)')
print "time period is",t2#prints number of orbits taken br stars for one orbit of planet
