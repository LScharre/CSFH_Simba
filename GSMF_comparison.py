#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:18:40 2024

@author: luciescharre
"""

import math
import caesar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.lines import Line2D


#%% 
model = 'm50n512'
size = 50
colors = ['orange','darkmagenta','dodgerblue', 'tomato',  'seagreen' ]

snaps = [50,78,104,151]


data_cat = #folder with sub folders 'nofb', 'noagn','nojet','nox','7jk' with respective Simba galaxy catalogues

#%% computation functions

# compute mass function
def mass_func(masses_lg,Vol_Mpc, step, bins):
        # create histgram from that octant octants 
        N, bin_edges = np.histogram(masses_lg,bins=bins)

        # compute the stellar mass function based on that 
        SMF = np.log10(N/(Vol_Mpc*step))
        
        return SMF


# DEFINING JACKKNIFE SAMPLING

def jknife_resample(galaxy_pos,galaxy_masses,L,bins,step):
    
    # define octants
    octants = [[0,0,0],[L/2,0,0],[0,L/2,0],[0,0,L/2]
              ,[L/2,L/2,0],[L/2,0,L/2],[0,L/2,L/2],[L/2,L/2,L/2]]
    
    SMFs = []
    
    
    for corner in octants:
        
        new_mask = []
        for pos in galaxy_pos:
            
            #check if the galaxy lies in the octant  
            if corner[0] <= pos[0] <= corner[0]+ L/2 and corner[1] <= pos[1] <= corner[1]+ L/2 and corner[2] <= pos[2] <= corner[2]+ L/2:
                
                #exclude current octant from the catalog
                new_mask.append(True)
                
            else:
                new_mask.append(False)
            
        # apply mask to galaxy masses to only include current octant 
        jk_mass = galaxy_masses[new_mask]
        
        jk_mass_lg = np.log10(jk_mass) 
    
        Vol = ((L/1000)**3)/8
        
        
        SMF = mass_func(jk_mass_lg, Vol,step,bins)               
        SMFs.append(SMF)
    
    # transpose to give a range of bin contents for the different octants 
    SMF_range = np.array(SMFs).T
    SMF_range  = np.ma.masked_invalid(SMF_range)
    
    var  = 1/math.sqrt(7)*np.ma.std(SMF_range, axis=1)     
    
    return var

  
def master_SMF(model,fb_fol,snap,size):
    infile = f'{data_cat}{fb_fol}/{model}_{snap:03d}.hdf5'
    sim = caesar.load(infile)
    h = sim.simulation.hubble_constant    
    z = sim.simulation.redshift
    Vol_Mpc = (size/h)**3
    if z<1:
        z_round =int(round(z))
    else:
        z_round =int(round(z)) 

    ################################################### PROCESSING AND PLOTTING
    # Collect the galaxy masses 
    galaxy_masses = [i.masses['stellar'] for i in sim.galaxies] 
    galaxy_masses=np.array(galaxy_masses)
    
    # take the logarithm of masses 
    galaxy_masses_lg = np.log10(galaxy_masses) 
    
    # create the bins between the min and max of the galaxy masses
    Nbins = 15
    bins = np.linspace(galaxy_masses_lg.min(),galaxy_masses_lg.max(),Nbins) 
    step = ((galaxy_masses_lg.max()-galaxy_masses_lg.min())/Nbins)
  
    # compute the SMF per volume
    SMF = mass_func(galaxy_masses_lg, Vol_Mpc, step, bins)

    # collect the positions of glaxies in comoving kiloparsec
    galaxy_pos = [np.array(i.pos) for i in sim.galaxies] 
    galaxy_pos = np.array(galaxy_pos)
    
    # define side lengthof box
    L=size*1000/h # kpc cm
    
    # call resampling function to find the cosmic variance
    var = jknife_resample(galaxy_pos, galaxy_masses, L,bins,step)
    
    # find the bincentres 
    bincen = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        bincen[i] = 0.5*(bins[i]+bins[i+1])
    
    return bincen, SMF, var, z_round     
        

#%% 
SMALL_SIZE = 8
MEDIUM_SIZE = 16
BIGGER_SIZE = 12
plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)            # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def GSMF_comp():
    colors = ["k","r","b","g"]
    
    fig, ax = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.set_size_inches(11,9.5)
    colors = ['orange','darkmagenta','dodgerblue', 'tomato',  'seagreen' ]
    fb_fols  = ['nofb','noagn','nojet','nox','7jk']  
    fb_types= [ 'no fb',  'stellar','AGN winds', 'jets','x-ray']  
    fb_types = ['No-feedback', 'No-AGN','No-jet','No-X-ray', 'Simba-50']

    snaps = [50,78,104,151]
        
    for i in range(len(snaps)): 
        snap = snaps[i]
        j=0
        
        for j in range(len(fb_types)):
            fb_type = fb_fols[j]
            fb_title = fb_types[j]
            color = colors[j]
        
            size =50
            model = 'm50n512'
            bincen, SMF, var, z_round = master_SMF(model,fb_type,snap, size)
            plot=ax.flat[i].plot(bincen,SMF,label = '%s' %(fb_title),color=color, lw=2)
            ax.flat[i].fill_between(bincen, SMF-var, SMF+var,facecolor= color, alpha = 0.3)
            at = AnchoredText('$z = %s$' %(z_round), prop=dict(size=14), frameon=True, loc='upper right')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.flat[i].add_artist(at)
            
            
            # plot the m100n1024 to test convergence
            if fb_type == '7jk':
                model = 'm100n1024'
                size = 100
                bincen, SMF, var, z_round = master_SMF(model,fb_type,snap,size)
                plot=ax.flat[i].plot(bincen,SMF,label = 'Simba-100',color='grey', lw=2)
                ax.flat[i].fill_between(bincen, SMF-var, SMF+var,facecolor= "grey", alpha = 0.3)
                
        
        ax.flat[i].set_ylim(-5.3,-0.75)
        ax.flat[i].set_xlim(8.7,12.9)
        ax.flat[i].tick_params(right=True, top=True)   

    ax.flat[0].set_xticklabels([])
    ax.flat[1].set_xticklabels([])
    
    ax.flat[1].set_yticklabels([])#, direction='inout')
    ax.flat[3].set_yticklabels([])#, direction='inout')

      
    ax.flat[2].set_xlabel("log ($M_{\star}\,/\,\mathrm{M_{\odot}}$)")
    ax.flat[3].set_xlabel("log ($M_{\star}\,/\,\mathrm{M_{\odot}}$)")
    ax.flat[0].set_ylabel("log ($\phi\,/\,\mathrm{Mpc^{-3}}$)")
    ax.flat[2].set_ylabel("log ($\phi\,/\,\mathrm{Mpc^{-3}}$)")
    
    ax.flat[3].legend(loc='lower left',fontsize =12, ncol = 1)

    plt.savefig('GSMF_comp.pdf'  , bbox_inches='tight', dpi=1200)
    
        

GSMF_comp() 

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
