"""
Monte-carlo estimation of pi
"""

import numpy as np
import random as r
import matplotlib.pyplot as plt
import os

params = {'font.family':'serif',
        'axes.labelsize':'small',
        'xtick.labelsize':'x-small',
        'ytick.labelsize':'x-small', 
        'lines.markersize': 2.5,
        'legend.fontsize':'small',
        'legend.title_fontsize':'small',
        'legend.fancybox': True,
        'legend.framealpha': 0.5,
        'legend.shadow': False,
        'legend.frameon': True,
        'grid.linestyle':'--',
        'grid.linewidth':'0.5',
        'lines.linewidth':'0.5'}
plt.rcParams.update(params)

def PiMonteCarlo(n):
    

    
    x = np.random.rand(n)
    y = np.random.rand(n)
    r = np.sqrt(np.add(np.power(x,2),np.power(y,2)))
    
    inside = len(np.where(r<1)[0])
    ratio = inside/n
    return ratio*4, x, y, r

n = 10000

pi_guess, x, y, r = PiMonteCarlo(1000)




squareX = [1,0,0,1,1]
squareY = [1,1,0,0,1]
circleX = (np.cos(np.pi*np.arange(n+1)/180))
circleY = (np.sin(np.pi*np.arange(n+1)/180))
indin = np.where(r<1)
indout = np.where(r>=1)


n = np.linspace(start = 10, stop = 1000000, num = 1000, dtype = int)
pis = []
for i in range(len(n)):
    pi, x,y,r = PiMonteCarlo(n[i])
    pis.append(pi)
pis = np.asarray(pis)


fig, [ax, ax1] = plt.subplots(1,2)
fig.set_size_inches(8, 4)
ax.plot(squareX,squareY,color='#000000')
ax.plot(circleX,circleY,color='#0000CC')
ax.scatter(x[indin], y[indin],  marker=".", s=10)
ax.scatter(x[indout], y[indout], marker=".", s=10)
ax.set_xlabel('x')
ax.set_ylabel('y')
#.title('Monte Carlo estimation of Pi')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax1.plot(pis)
ax1.set_xlabel('number of samples drawn')
ax1.set_ylabel('pi estimate')
plt.tight_layout()

fig.savefig(os.path.join(os.environ['USERPROFILE'] + r"\Dropbox\General Python\MonteCarloPi.png"), format = 'png')

