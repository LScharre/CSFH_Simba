#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:55:35 2024

@author: luciescharre
"""

import caesar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from matplotlib.offsetbox import AnchoredText
import scipy.stats as stats

fb_fols  = ["nofb",'noagn','nojet','nox','7jk']  
fb_types = [ 'No-feedback','No-AGN','No-jet','No-X-ray', 'Simba-50']
colors = ['orange','darkmagenta','dodgerblue', 'tomato',  'seagreen' ]

snaps = [151]  
model = 'm50n512'
size = 50

MEDIUM_SIZE = 15
BIGGER_SIZE = 12
plt.rc('font', size=10)         # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=10)            # legend fontsize
plt.rc('figure', titlesize=10)  # fontsize of the figure title

for j in range(len(snaps)):
    fig = plt.figure()
    plt.subplots_adjust(wspace=0, hspace=0)
    #fig.set_size_inches(11,9)

    n=0
    for i in range(len(fb_types)):
    
        fb_type = fb_types[i]
        fb_fol = fb_fols[i]
    
        snap = snaps[j]
        infile = f'/Users/luciescharre/Downloads/MPhys/data_scripts polished/cats/{fb_fol}/{model}_{snap:03d}.hdf5'
        sim = caesar.load(infile)
        
        h = sim.simulation.hubble_constant    
        z = sim.simulation.redshift
        z_round = round(z) if z > 1.99 or z > 0 else round(z, 1)
        
        mH_sigma = np.array([i.velocity_dispersions['stellar'] for i in sim.galaxies if i.central==1])
        mH_BH = np.array([i.masses['bh'] for i in sim.galaxies if i.central==1])
        mH_stellar = np.array([i.masses['stellar'] for i in sim.galaxies if i.central==1])
        SFR = np.array([i.halo.sfr for i in sim.galaxies if i.central==1])
        sSFR = SFR / mH_stellar
        
        log_mH_stellar = np.log10(mH_stellar)
        log_mH_BH = np.log10(mH_BH)
        
        bins = np.linspace(9.5, log_mH_stellar[mH_stellar>0].max(), 20)
        #bins = np.linspace(log_mH_stellar[mH_stellar>0].min(), log_mH_stellar[mH_stellar>0].max(), 20)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        median_BH_mass = []
        perc16_BH_mass = []
        perc84_BH_mass = []
        
        for b1, b2 in zip(bins[:-1], bins[1:]):
            in_bin = (log_mH_stellar >= b1) & (log_mH_stellar < b2)
            if np.sum(in_bin) > 0:
                median_BH_mass.append(np.median(log_mH_BH[in_bin]))
                perc16_BH_mass.append(np.percentile(log_mH_BH[in_bin], 16))
                perc84_BH_mass.append(np.percentile(log_mH_BH[in_bin], 84))
            else:
                median_BH_mass.append(np.nan)
                perc16_BH_mass.append(np.nan)
                perc84_BH_mass.append(np.nan)

        median_BH_mass = np.array(median_BH_mass)
        perc16_BH_mass = np.array(perc16_BH_mass)
        perc84_BH_mass = np.array(perc84_BH_mass)
        
        plt.plot(bin_centers, median_BH_mass, color=colors[i], label = fb_types[i])
        plt.fill_between(bin_centers, perc16_BH_mass, perc84_BH_mass, color=colors[i], alpha=0.3)

        print(log_mH_stellar[(log_mH_stellar<9.5)&(log_mH_BH>0)].min())
        print(log_mH_BH[(log_mH_stellar<9.5)&(log_mH_BH>0)].min())
        print()
        
        log_Mstar_range = 10**bin_centers
        
        log_M_BH =(8.20) +(1.12) * np.log10(log_Mstar_range/1e11)
        log_M_BH =-(0.266) -0.484 * (log_Mstar_range+24.21)+9
        
        alpha = np.log10(0.49)
        beta = 1.16
        log_Mstar_range = np.linspace(9.5, 13, 20)
        log_M_BH = 9 +alpha + beta*(log_Mstar_range-11)
        
        plt.ylim(4.2, 10)
        plt.xlim(9.5, 12.4)
        plt.ylim(5.5, 10.5)
        plt.xlim(9.5, 13)
        
        """at = AnchoredText(f'{fb_type}', prop=dict(size=14), frameon=True, loc='lower right')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        plt.add_artist(at)"""
        plt.tick_params(axis="y", direction="inout", right=True, top=True)
        plt.tick_params(axis="x", direction="inout", right=True, top=True)
        #plt.grid(visible=True)
     
    #plt.plot(log_Mstar_range, log_M_BH, color="k", label = "Kormendy & Ho 2013")
    plt.legend(fontsize =12)
       
    
    plt.ylabel('$\\mathrm{log}(\\,M_\\mathrm{BH}\\,/\\,{\\mathrm{M_{\\odot}}}\\,)$')
    
    plt.xlabel('$\\mathrm{log}(\\,M_{\\mathrm{\\star}} \\, /\\,\\mathrm{M_{\\odot}}\\,$)')
    
    
    #plt.gcf().text(0.78, 0.8, f'$z = {int(z_round)}$', fontsize=25)
    fig.savefig(f'/Users/luciescharre/Downloads/MPhys/data_scripts polished/BH_mass_stellar_mass_{z_round}.pdf', bbox_inches='tight', dpi=1200)
