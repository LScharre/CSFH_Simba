#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:00:48 2024

@author: luciescharre
"""


import numpy as np
import matplotlib.pyplot as plt
import caesar
from astropy.cosmology import WMAP9 as cosmo
from matplotlib.offsetbox import AnchoredText
import h5py

model = "m50n512"
size = 50


# local files
fb_fols  = ['nofb','noagn','nojet','nox','7jk']  
fb_types = ['No-feedback', 'No-AGN','No-jet','No-X-ray', 'Simba-50']

snaps = [36, 50, 62, 78, 84, 90, 97, 104, 125, 137, 142, 151]


data_cat = #folder with sub folders 'nofb' 'noagn','nojet','nox','7jk' with respective Simba galaxy catalogues

#%% computing the bins

def SFR_halobins(model, fb_fol,server_fb_fol, snap, size):
    # LOADING CATALOGUES & PRINTING IMPORTANT INFO
    
    infile = f'{data_cat}{fb_fol}/{model}_{snap:03d}.hdf5'
    sim = caesar.load(infile)
    z = sim.simulation.redshift

     
    mH_tot = np.array([i.halo.masses['total']
                      for i in sim.galaxies if i.central == 1])
    mH_stellar = np.array([i.halo.masses['stellar']
                          for i in sim.galaxies if i.central == 1])
    hal_SFR = np.array([i.halo.sfr for i in sim.galaxies if i.central == 1])

    mH_tot = mH_tot
    hal_SFR = hal_SFR

        
    mH_tot_lg  = np.log10(mH_tot)  

    mH_11 = []
    mH_12 = []
    mH_13 = []
    mH_14 = []

    # appends to lists of indices belonging to relevant mass bins
    for mi in range(len(mH_tot_lg)):

        if 10.5 < mH_tot_lg[mi] < 11.5:
            mH_11.append(mi)

        elif 11.5 < mH_tot_lg[mi] < 12.5:
            mH_12.append(mi)

        elif 12.5 < mH_tot_lg[mi] < 13.5:
            mH_13.append(mi)

        elif 13.5 < mH_tot_lg[mi] < 14.5:
            mH_14.append(mi)

    mH_bins_ids = [mH_11, mH_12, mH_13, mH_14]
    # print(mH_11)

    SFR_bins_med = [[], [], [], []]
    sSFR_bins_med = [[], [], [], []]
    nSFR_bins_med = [[], [], [], []]

    SFR_bins_16 = [[], [], [], []]
    sSFR_bins_16 = [[], [], [], []]
    nSFR_bins_16 = [[], [], [], []]

    SFR_bins_84 = [[], [], [], []]
    sSFR_bins_84 = [[], [], [], []]
    nSFR_bins_84 = [[], [], [], []]

    # using the indices, I can then find the median and percentiles of the SFR
    for j in range(len(mH_bins_ids)):
        
        mH_bin = mH_bins_ids[j]
        SFR_temp = []
        sSFR_temp = []
        nSFR_temp = []

        # collect the SFR
        for i in mH_bin:
            SFR_temp.append(hal_SFR[i])
            sSFR_temp.append(hal_SFR[i]/mH_stellar[i])
            nSFR_temp.append(hal_SFR[i]/mH_tot[i])
            

        # compute the percentiles

        if len(SFR_temp) == 0:
            SFR_bins_med[j] = float("nan")
            sSFR_bins_med[j] = float("nan")
            nSFR_bins_med[j] = float("nan")
            SFR_bins_16[j] = float("nan")
            sSFR_bins_16[j] = float("nan")
            nSFR_bins_16[j] = float("nan")
            SFR_bins_84[j] = float("nan")
            sSFR_bins_84[j] = float("nan")
            nSFR_bins_84[j] = float("nan")

        else:
            SFR_bins_med[j] = np.percentile(SFR_temp, 50)
            sSFR_bins_med[j] = np.percentile(sSFR_temp, 50)
            nSFR_bins_med[j] = np.percentile(nSFR_temp, 50)

            SFR_bins_16[j] = np.percentile(SFR_temp, 16)
            sSFR_bins_16[j] = np.percentile(sSFR_temp, 16)
            nSFR_bins_16[j] = np.percentile(nSFR_temp, 16)

            SFR_bins_84[j] = np.percentile(SFR_temp, 84)
            sSFR_bins_84[j] = np.percentile(sSFR_temp, 84)
            nSFR_bins_84[j] = np.percentile(nSFR_temp, 84)

    SFR_data = [SFR_bins_med, SFR_bins_16, SFR_bins_84]
    sSFR_data = [sSFR_bins_med, sSFR_bins_16, sSFR_bins_84]
    nSFR_data = [nSFR_bins_med, nSFR_bins_16, nSFR_bins_84]

    age = cosmo.age(z).value
    return z, age, SFR_data, sSFR_data, nSFR_data


#%% plotting in subplots
SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 12
plt.rc('font', size=10)         # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=10)            # legend fontsize
plt.rc('figure', titlesize=10)  # fontsize of the figure title


def halobins_plot(model, fb_fols,server_fb_fols,  fb_types,snaps,size):
    
    # loop calling the master_SMF function
    fig, ax = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.set_size_inches(10,9)


    for i in range(len(fb_fols)):
        fb_type = fb_types[i]
        fb_fol = fb_fols[i]
        server_fol = server_fb_fols[i]

        redshifts = []
        ages = []

        nSFR_bins_z = []    
        nSFR_bins_16_z = []
        nSFR_bins_84_z = []

        for snap in snaps:
            z, age, SFR_data, sSFR_data, nSFR_data = SFR_halobins(
                model, fb_fol, server_fol,snap, size)
    
            redshifts.append(z)
            ages.append(age)
    
            nSFR_bins_z.append(nSFR_data[0])
            nSFR_bins_16_z.append(nSFR_data[1])
            nSFR_bins_84_z.append(nSFR_data[2])
    
        # Transpose arrays to gather values for the bins together
        nSFR_bins_z = np.array(nSFR_bins_z).T
        nSFR_bins_16_z = np.array(nSFR_bins_16_z).T
        nSFR_bins_84_z = np.array(nSFR_bins_84_z).T

        alpha =0.2
        lw=2
        line1 = ax.flat[3].plot(ages,np.log10( nSFR_bins_z[3]), color=colors[i], lw=lw)
        ax.flat[3].fill_between(
            ages, np.log10(nSFR_bins_16_z[3]), np.log10(nSFR_bins_84_z[3]), facecolor=colors[i], alpha=alpha)

        line2 = ax.flat[2].plot(ages, np.log10(nSFR_bins_z[2]), color=colors[i], lw=lw)
        ax.flat[2].fill_between(
            ages, np.log10(nSFR_bins_16_z[2]), np.log10(nSFR_bins_84_z[2]), facecolor=colors[i], alpha=alpha)

        line3 =ax.flat[1].plot(ages, np.log10(nSFR_bins_z[1]), color=colors[i], lw=lw)
        ax.flat[1].fill_between(
            ages, np.log10(nSFR_bins_16_z[1]), np.log10(nSFR_bins_84_z[1]), facecolor=colors[i], alpha=alpha)

        line4 =ax.flat[0].plot(ages, np.log10(nSFR_bins_z[0]), color=colors[i], lw=lw)
        ax.flat[0].fill_between(
            ages, np.log10(nSFR_bins_16_z[0]), np.log10(nSFR_bins_84_z[0]), facecolor=colors[i], alpha=alpha)
        
        lines = [line1,line2,line3,line4]
        labels = ['$10^{10.5}\leq M_\mathrm{H} < 10^{11.5} \,\mathrm{M}_{\odot}$','$10^{11.5}\leq M_\mathrm{H} < 10^{12.5} \,\mathrm{M}_{\odot}$','$10^{12.5}\leq M_\mathrm{H} < 10^{13.5} \,\mathrm{M}_{\odot}$','$10^{13.5}\leq M_\mathrm{H} \leq 10^{14.5} \,\mathrm{M}_{\odot}$']
        
        #ax.flat[i].set_ylim(8e-15, 2e-10)  # nSFR
        #ax.flat[i].set_yscale('log')
        
        if i<4:
            ax.flat[i].set_ylim(-13.7, -9.7)
            ax.flat[i].tick_params(right=True, top=True)   
            ax.flat[i].tick_params(axis="y",direction="inout")
            ax.flat[i].tick_params(axis="x",direction="inout")
        
            at = AnchoredText('%s' %(labels[i]), prop=dict(size=12), frameon=True, loc='upper right')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.flat[i].add_artist(at)

            
            ax.flat[i].set_xlim(0.5,14)
            ax.flat[i].set_xticks([2,4,6,8,10,12,14])
            

    line1 = ax.flat[0].plot([], [], color=colors[0], label =fb_types[0])

    line2 = ax.flat[0].plot([], [], color=colors[1], label =fb_types[1])
       
    line3 =ax.flat[0].plot([], [], color=colors[2], label =fb_types[2])
       
    line4 =ax.flat[0].plot([], [],color=colors[3], label =fb_types[3])
    line5 =ax.flat[0].plot([], [],color=colors[4], label=fb_types[4])
    ax.flat[0].legend(loc= "lower center", ncol=2)
    
 
    ax.flat[0].set_xticklabels([])#, direction='inout')
    ax.flat[1].set_xticklabels([])#, direction='inout')   
   
    ax.flat[2].set_xlabel('Cosmic Time [Gyr]')
    ax.flat[3].set_xlabel('Cosmic Time [Gyr]')
    
    ax.flat[0].set_ylabel('log (nSFR / $\mathrm{yr}^{-1}$)')
    ax.flat[0].set_yticks([-13,-12,-11,-10])#, direction='inout')
    ax.flat[1].set_yticks([-13,-12,-11,-10],labels =[])#, direction='inout')
    ax.flat[2].set_ylabel('log (nSFR / $\mathrm{yr}^{-1}$)')
    ax.flat[2].set_yticks([-13,-12,-11,-10])#, direction='inout')
    ax.flat[3].set_yticks([-13,-12,-11,-10],labels =[])#, direction='inout')
    #ax.flat[4].set_ylabel('log (nSFR / $\mathrm{yr}^{-1}$)')
   # ax.flat[5].set_yticklabels([])#, direction='inout')
    #ax.flat[5].set_ylim(ax.flat[4].get_ylim())
    #ax.flat[-1].axis('off')
    redshift_ticks = [0, 1, 2, 4, 6]
    age_ticks = cosmo.age(redshift_ticks).value
    
    ax_up0 = ax.flat[0].twiny()
    ax_up0.set_xlim(ax.flat[0].get_xlim())
    ax_up0.set_xticks(age_ticks)
    ax_up0.set_xlabel('$z$')
    ax_up0.set_xticklabels(redshift_ticks)
    
    ax_up1 = ax.flat[1].twiny()
    ax_up1.set_xlim(ax.flat[1].get_xlim())
    ax_up1.set_xticks(age_ticks)
    ax_up1.set_xlabel('$z$')
    ax_up1.set_xticklabels(redshift_ticks)
    
    ax_up1 = ax.flat[2].twiny()
    ax_up1.set_xlim(ax.flat[2].get_xlim())
    ax_up1.set_xticks(age_ticks, direction="in")
    ax_up1.tick_params(direction="in")
    ax_up1.set_xticklabels([])
    
    ax_up1 = ax.flat[3].twiny()
    ax_up1.set_xlim(ax.flat[3].get_xlim())
    ax_up1.set_xticks(age_ticks, direction="in")
    ax_up1.tick_params(direction="in")
    ax_up1.set_xticklabels([])
    
    plt.savefig('nSFR_halobins_subplots.pdf', bbox_inches='tight', dpi =1200)
    
    
     
fb_fols  = ['nofb','noagn','nojet','nox','7jk']   
fb_types = ['No-feedback', 'No-AGN','No-jet','No-X-ray', 'Simba-50']
colors = ['orange','darkmagenta','dodgerblue', 'tomato',  'seagreen' ]
snaps = [36, 50, 62, 78, 84, 90, 97, 104, 125, 137, 142, 151]

size=50
halobins_plot(model, fb_fols, server_fb_fols, fb_types,snaps,size) 