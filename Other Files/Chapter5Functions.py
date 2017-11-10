GRAVITATION = 10 #Gravitational constant G
MASS = 50        #Mass of accreting star


def AngularVelocity(R):
    angVel = ((GRAVITATION * MASS)/R^3)**0.5
    return angVel
    
def circularVelocity(R):
  circVel = R*AngularVelocity(R)
  return circVel
  
for r in range(1,10):
  print('For a radius of', r, 'Star mass', MASS, 'the angular veloccity is', AngularVelocity(r))
  print('The circular velocity is', circularVelocity(r))
  print('-------')
  
  
print('A larger radius creates a smaller angular velocity but a larger circular velocity')
print('Both of these are of course independent of the radius of the star R*')