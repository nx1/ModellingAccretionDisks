# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import discfunc as df


def create_polar (R):
    R_new = []
    Theta =  np.tile(np.linspace(0, 2*np.pi, 100), len(R))
    print(len(R))
    for i in range(len(R)):
        
        for j in range(100):
            R_new.append(R[i])

    return R_new, Theta


r = df.create_disc(5,2.0,8.0,1.0)
r_new, theta = create_polar(r)


ax = plt.subplot(111, projection='polar')

ax.plot(theta, r_new)

#ax.set_rmax(max(r))
ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
ax.grid(True)

ax.set_title("A line plot on a polar axis", va='bottom')
plt.show()
