#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:55:39 2024

@author: luciescharre
"""
import caesar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from scipy.ndimage.filters import gaussian_filter
from matplotlib.offsetbox import AnchoredText

model = 'm50n512'
size = 50
data_cat = #folder with sub folders 'noagn','nojet','nox','7jk' with respective Simba galaxy catalogues

#%%

MEDIUM_SIZE = 15
BIGGER_SIZE = 12
plt.rc('font', size=10)         # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=10)            # legend fontsize
plt.rc('figure', titlesize=10)  # fontsize of the figure title


def sSFR_mStar_subplots(model, snaps, size, fb_fols, fb_types):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.set_size_inches(10,13)
        n =0
        for i in range(len(fb_types)):
            fb_type = fb_types[i]
            fb_fol = fb_fols[i]
            for j in range(len(snaps)):
                snap = snaps[j]
                
                infile = f'{data_cat}{fb_fol}/{model}_%03d.hdf5' %(snap)
                sim = caesar.load(infile)
                
                h = sim.simulation.hubble_constant    
                z = sim.simulation.redshift
                z_round =round(z)

                mH_tot = np.array([i.halo.masses['total'] for i in sim.galaxies if i.central==1])
                mH_stellar = np.array([i.masses['stellar'] for i in sim.galaxies if i.central==1])
                mH_BH = np.array([i.masses['bh'] for i in sim.galaxies if i.central==1])
                SFR = np.array([i.halo.sfr for i in sim.galaxies if i.central==1])
                bh_fedd= np.array([i.halo.bh_fedd for i in sim.galaxies if i.central==1])  
                mH_gas = np.array([i.masses['gas'] for i in sim.galaxies if i.central==1])

                
                mH_tot = np.log10(mH_tot[mH_BH>0])
                
                mH_stellar = mH_stellar[mH_BH>0]
                SFR =SFR[mH_BH>0]
                bh_fedd= bh_fedd[mH_BH>0]
                mH_gas = mH_gas[mH_BH>0]
                mH_BH = mH_BH[mH_BH>0]
                bh_fedd[bh_fedd==0]=1e-8
                
                BHm_frac = np.log10(mH_BH/mH_stellar)
                gas_frac = mH_gas/mH_stellar
                sSFR = np.log10(SFR/mH_stellar)
                mH_stellar = np.log10(mH_stellar)

                at = AnchoredText('%s' %(fb_type), prop=dict(size=14), frameon=True, loc='lower right')
                at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
                ax.flat[i].add_artist(at)
                plt.gcf().text(0.15, 0.16, '$z = %s$' %(int(z_round)), fontsize=25)

                vmin = 1e-4
                vmax = 1
                cmap = 'jet_r'
                clabel = '$f_{edd}$'
                
                lw =2
                ls ="dashed"

                alpha=1

                s_1 = 1
                s_2 = 1
                s_3 = 5
                s_4 = 30
                s_5 = 100
                
                if snap==78:
                    wind_mask = np.ma.masked_inside(mH_BH, 0, 10**7.5).mask | (bh_fedd>0.2)
                    jet_mask = (mH_BH >  10**7.5) & (bh_fedd<0.2) & (gas_frac>0.2)
                    xray_mask = (mH_BH >  10**7.5) & (bh_fedd<0.2) & (gas_frac<0.2)

                    # WINDS
                    plot1 = ax.flat[n].scatter(mH_stellar[wind_mask & (BHm_frac < -4)],        sSFR[wind_mask & (BHm_frac < -4)],         c= bh_fedd[wind_mask & (BHm_frac < -4)],           norm=mlp.colors.LogNorm(vmin,vmax), s=s_1,cmap=cmap, alpha =alpha)
                    plot = ax.flat[n].scatter(mH_stellar[wind_mask & (BHm_frac < -3) & (BHm_frac > -4)],  sSFR[wind_mask & (BHm_frac < -3) & (BHm_frac > -4)],  c= bh_fedd[wind_mask & (BHm_frac < -3) & (BHm_frac > -4)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_2,    cmap=cmap, alpha =alpha, edgecolor =None)
                    plot = ax.flat[n].scatter(mH_stellar[wind_mask & (BHm_frac < -2.5) & (BHm_frac > -3)],  sSFR[wind_mask & (BHm_frac < -2.5) & (BHm_frac > -3)],  c= bh_fedd[wind_mask & (BHm_frac < -2.5) & (BHm_frac > -3)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_3,    cmap=cmap, alpha =alpha, edgecolor =None)
                    plot = ax.flat[n].scatter(mH_stellar[wind_mask & (BHm_frac < -2) & (BHm_frac > -2.5)],  sSFR[wind_mask & (BHm_frac < -2) & (BHm_frac > -2.5)], c= bh_fedd[wind_mask & (BHm_frac < -2) & (BHm_frac > -2.5)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_4,    cmap=cmap, alpha =alpha, edgecolor =None)
                    plot = ax.flat[n].scatter(mH_stellar[wind_mask & (BHm_frac < -1) & (BHm_frac > -2)],  sSFR[wind_mask & (BHm_frac < -1) & (BHm_frac > -2)], c= bh_fedd[wind_mask & (BHm_frac < -1) & (BHm_frac > -2)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_5,    cmap=cmap, alpha =alpha, edgecolor =None)

                    # JETS 
                    ec = "k"
                    lw=0.1
                    plot = ax.flat[n].scatter(mH_stellar[jet_mask & (BHm_frac < -4)],        sSFR[jet_mask & (BHm_frac < -4)],         c= bh_fedd[jet_mask & (BHm_frac < -4)],           norm=mlp.colors.LogNorm(vmin,vmax), s=s_1,cmap=cmap, marker = 's', alpha =alpha, edgecolor =ec,linewidths=lw)
                    plot = ax.flat[n].scatter(mH_stellar[jet_mask & (BHm_frac < -3) & (BHm_frac > -4)],  sSFR[jet_mask & (BHm_frac < -3) & (BHm_frac > -4)],  c= bh_fedd[jet_mask & (BHm_frac < -3) & (BHm_frac > -4)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_2,cmap=cmap, marker = 's', alpha =alpha, edgecolor =ec,linewidths=lw)
                    plot = ax.flat[n].scatter(mH_stellar[jet_mask & (BHm_frac < -2.5) & (BHm_frac > -3)],  sSFR[jet_mask & (BHm_frac < -2.5) & (BHm_frac > -3)],  c= bh_fedd[jet_mask & (BHm_frac < -2.5) & (BHm_frac > -3)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_3,cmap=cmap, marker = 's', alpha =alpha, edgecolor =ec,linewidths=lw)
                    plot = ax.flat[n].scatter(mH_stellar[jet_mask & (BHm_frac < -2) & (BHm_frac > -2.5)],  sSFR[jet_mask & (BHm_frac < -2) & (BHm_frac > -2.5)],  c= bh_fedd[jet_mask & (BHm_frac < -2) & (BHm_frac > -2.5)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_4,cmap=cmap, marker = 's', alpha =alpha, edgecolor =ec,linewidths=lw)
                    plot = ax.flat[n].scatter(mH_stellar[jet_mask & (BHm_frac < -1) & (BHm_frac > -2)],  sSFR[jet_mask & (BHm_frac < -1) & (BHm_frac > -2)], c= bh_fedd[jet_mask & (BHm_frac < -1) & (BHm_frac > -2)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_5,  cmap=cmap, marker = 's', alpha =alpha, edgecolor =ec,linewidths=lw)
                   
                    # X-RAY 
                    alpha=0.8
                    plot = ax.flat[n].scatter(mH_stellar[xray_mask & (BHm_frac < -4)],        sSFR[xray_mask & (BHm_frac < -4)],         c= bh_fedd[xray_mask & (BHm_frac < -4)],           norm=mlp.colors.LogNorm(vmin,vmax), s=s_1,cmap=cmap, marker = 'v', alpha =alpha, edgecolor =None)
                    plot = ax.flat[n].scatter(mH_stellar[xray_mask & (BHm_frac < -3) & (BHm_frac > -4)],  sSFR[xray_mask & (BHm_frac < -3) & (BHm_frac > -4)],  c= bh_fedd[xray_mask & (BHm_frac < -3) & (BHm_frac > -4)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_2,cmap=cmap, marker = 'v', alpha =alpha, edgecolor =None)
                    plot = ax.flat[n].scatter(mH_stellar[xray_mask & (BHm_frac < -2.5) & (BHm_frac > -3)],  sSFR[xray_mask & (BHm_frac < -2.5) & (BHm_frac > -3)], c= bh_fedd[xray_mask & (BHm_frac < -2.5) & (BHm_frac > -3)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_3,cmap=cmap, marker = 'v', alpha =alpha, edgecolor =None)
                    plot = ax.flat[n].scatter(mH_stellar[xray_mask & (BHm_frac < -2) & (BHm_frac > -2.5)],  sSFR[xray_mask & (BHm_frac < -2) & (BHm_frac > -2.5)],  c= bh_fedd[xray_mask & (BHm_frac < -2) & (BHm_frac > -2.5)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_4,cmap=cmap, marker = 'v', alpha =alpha, edgecolor =None)
                    plot = ax.flat[n].scatter(mH_stellar[xray_mask & (BHm_frac < -1) & (BHm_frac > -2)],  sSFR[xray_mask & (BHm_frac < -1) & (BHm_frac > -2)], c= bh_fedd[xray_mask & (BHm_frac < -1) & (BHm_frac > -2)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_5,cmap=cmap, marker = 'v', alpha =alpha, edgecolor =None)
                   
                    
                    ax.flat[n].set_ylim(5e-13,1e-8)
                    ax.flat[n].set_xlim(2e9,1e12)
                    
                    ax.flat[n].set_ylim(np.log10(5e-13),-8)
                    ax.flat[n].set_ylim(-11.8,-8)
                    ax.flat[n].set_xlim(9.3,12)
 
                    if n==0:
                        size_array = [1,5,30,60]
                        labels = ['$M_\mathrm{BH}/M_{\star} < 10^{-3}$', '$M_\mathrm{BH}/M_{\star} < 10^{-2.5}$', '$M_\mathrm{BH}/M_{\star} < 10^{-2}$', '$M_\mathrm{BH}/M_{\star} < 10^{-1}$']
                        #markers = ["o","o","o"]
                        for size, label in zip(size_array, labels):
                            ax.flat[n].scatter([], [], s=size, label=label, marker='o', color ="k")
                        ax.flat[n].legend(loc= "lower left",framealpha=1, frameon =True, edgecolor="white")
                    
                    elif n==1:
                        size_array = [30,30,30]
                        labels = ['winds',"jets","x-ray"]
                        markers = ["o","s","v"]
                        for size, label, marker in zip(size_array, labels, markers):
                            ax.flat[n].scatter([], [], s=size, label=label, marker=marker, color ="k")
                        ax.flat[n].legend(loc= "lower left",framealpha=1,title="Eligible for:", frameon =True, edgecolor="white")

                    
                if snap ==151:
                    cmap = "jet_r"
                    wind_mask = np.ma.masked_inside(mH_BH, 0, 10**7.5).mask | (bh_fedd>0.2)
                    jet_mask = (mH_BH >  10**7.5) & (bh_fedd<0.2) & (gas_frac>0.2)
                    xray_mask = (mH_BH >  10**7.5) & (bh_fedd<0.2) & (gas_frac<0.2)
                    
                    # WINDS
                    plot1 = ax.flat[n].scatter(mH_stellar[wind_mask & (BHm_frac < -4)],        sSFR[wind_mask & (BHm_frac < -4)],         c= bh_fedd[wind_mask & (BHm_frac < -4)],           norm=mlp.colors.LogNorm(vmin,vmax), s=s_1,cmap=cmap, alpha =alpha)
                    plot = ax.flat[n].scatter(mH_stellar[wind_mask & (BHm_frac < -3) & (BHm_frac > -4)],  sSFR[wind_mask & (BHm_frac < -3) & (BHm_frac > -4)],  c= bh_fedd[wind_mask & (BHm_frac < -3) & (BHm_frac > -4)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_2,    cmap=cmap, alpha =alpha, edgecolor =None)
                    plot = ax.flat[n].scatter(mH_stellar[wind_mask & (BHm_frac < -2.5) & (BHm_frac > -3)],  sSFR[wind_mask & (BHm_frac < -2.5) & (BHm_frac > -3)],  c= bh_fedd[wind_mask & (BHm_frac < -2.5) & (BHm_frac > -3)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_3,    cmap=cmap, alpha =alpha, edgecolor =None)
                    plot = ax.flat[n].scatter(mH_stellar[wind_mask & (BHm_frac < -2) & (BHm_frac > -2.5)],  sSFR[wind_mask & (BHm_frac < -2) & (BHm_frac > -2.5)], c= bh_fedd[wind_mask & (BHm_frac < -2) & (BHm_frac > -2.5)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_4,    cmap=cmap, alpha =alpha, edgecolor =None)
                    plot = ax.flat[n].scatter(mH_stellar[wind_mask & (BHm_frac < -1) & (BHm_frac > -2)],  sSFR[wind_mask & (BHm_frac < -1) & (BHm_frac > -2)], c= bh_fedd[wind_mask & (BHm_frac < -1) & (BHm_frac > -2)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_5,    cmap=cmap, alpha =alpha, edgecolor =None)
                    
                    # JETS of three sizes
                    ec = "k"
                    lw=0.1
                    plot = ax.flat[n].scatter(mH_stellar[jet_mask & (BHm_frac < -4)],        sSFR[jet_mask & (BHm_frac < -4)],         c= bh_fedd[jet_mask & (BHm_frac < -4)],           norm=mlp.colors.LogNorm(vmin,vmax), s=s_1,cmap=cmap, marker = 's', alpha =alpha, edgecolor =ec,linewidths=lw)
                    plot = ax.flat[n].scatter(mH_stellar[jet_mask & (BHm_frac < -3) & (BHm_frac > -4)],  sSFR[jet_mask & (BHm_frac < -3) & (BHm_frac > -4)],  c= bh_fedd[jet_mask & (BHm_frac < -3) & (BHm_frac > -4)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_2,cmap=cmap, marker = 's', alpha =alpha, edgecolor =ec,linewidths=lw)
                    plot = ax.flat[n].scatter(mH_stellar[jet_mask & (BHm_frac < -2.5) & (BHm_frac > -3)],  sSFR[jet_mask & (BHm_frac < -2.5) & (BHm_frac > -3)],  c= bh_fedd[jet_mask & (BHm_frac < -2.5) & (BHm_frac > -3)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_3,cmap=cmap, marker = 's', alpha =alpha, edgecolor =ec,linewidths=lw)
                    plot = ax.flat[n].scatter(mH_stellar[jet_mask & (BHm_frac < -2) & (BHm_frac > -2.5)],  sSFR[jet_mask & (BHm_frac < -2) & (BHm_frac > -2.5)],  c= bh_fedd[jet_mask & (BHm_frac < -2) & (BHm_frac > -2.5)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_4,cmap=cmap, marker = 's', alpha =alpha, edgecolor =ec,linewidths=lw)
                    plot = ax.flat[n].scatter(mH_stellar[jet_mask & (BHm_frac < -1) & (BHm_frac > -2)],  sSFR[jet_mask & (BHm_frac < -1) & (BHm_frac > -2)], c= bh_fedd[jet_mask & (BHm_frac < -1) & (BHm_frac > -2)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_5,  cmap=cmap, marker = 's', alpha =alpha, edgecolor =ec,linewidths=lw)
                   
                    # X-RAY of three sizes
                    alpha=0.8
                    plot = ax.flat[n].scatter(mH_stellar[xray_mask &  (BHm_frac < -4) ],        sSFR[xray_mask &  (BHm_frac < -4)],         c= bh_fedd[xray_mask &  (BHm_frac < -4)],           norm=mlp.colors.LogNorm(vmin,vmax), s=s_1,cmap=cmap, marker = 'v', alpha =alpha)
                    plot = ax.flat[n].scatter(mH_stellar[xray_mask &  (BHm_frac < -3) & (BHm_frac > -4)],  sSFR[xray_mask &  (BHm_frac < -3) & (BHm_frac > -4)],  c= bh_fedd[xray_mask &  (BHm_frac < -3) & (BHm_frac > -4)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_2,cmap=cmap, marker = 'v', alpha =alpha)
                    plot = ax.flat[n].scatter(mH_stellar[xray_mask &  (BHm_frac < -2.5) & (BHm_frac > -3)],  sSFR[xray_mask &  (BHm_frac < -2.5) & (BHm_frac > -3)], c= bh_fedd[xray_mask &  (BHm_frac < -2.5) & (BHm_frac > -3)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_3,cmap=cmap, marker = 'v', alpha =alpha)
                    plot = ax.flat[n].scatter(mH_stellar[xray_mask &  (BHm_frac < -2) & (BHm_frac > -2.5)],  sSFR[xray_mask &  (BHm_frac < -2) & (BHm_frac > -2.5)],  c= bh_fedd[xray_mask &  (BHm_frac < -2) & (BHm_frac > -2.5)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_4,cmap=cmap, marker = 'v', alpha =alpha)
                    plot = ax.flat[n].scatter(mH_stellar[xray_mask &  (BHm_frac < -1) & (BHm_frac > -2)],  sSFR[xray_mask &  (BHm_frac < -1) & (BHm_frac > -2)], c= bh_fedd[xray_mask &  (BHm_frac < -1) & (BHm_frac > -2)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_5,cmap=cmap, marker = 'v', alpha =alpha)
                      
                    ax.flat[n].set_ylim(5e-13,1e-8)
                    ax.flat[n].set_xlim(2e9,1e12)
                    
                    ax.flat[n].set_ylim(np.log10(5e-13),-8)
                    ax.flat[n].set_ylim(-13,-8.7)
                    ax.flat[n].set_xlim(9.3,12)
                    
                    if n==0:
                        size_array = [1,5,30,60]
                        labels = ['$M_\mathrm{BH}/M_{\star} < 10^{-3}$', '$M_\mathrm{BH}/M_{\star} < 10^{-2.5}$', '$M_\mathrm{BH}/M_{\star} < 10^{-2}$', '$M_\mathrm{BH}/M_{\star} < 10^{-1}$']
                        for size, label in zip(size_array, labels):
                            ax.flat[n].scatter([], [], s=size, label=label, marker='o', color ="k")
                        ax.flat[n].legend(loc= "lower left",framealpha=1, frameon =True, edgecolor="white")
                    
                    elif n==1:
                        size_array = [30,30,30]
                        labels = ['winds',"jets","x-ray"]
                        markers = ["o","s","v"]
                        for size, label, marker in zip(size_array, labels, markers):
                            ax.flat[n].scatter([], [], s=size, label=label, marker=marker, color ="k")
                        ax.flat[n].legend(loc= "lower left",framealpha=1,title="eligible for:", frameon =True, edgecolor="white")
                    
            ax.flat[n].tick_params(right=True, top=True)   
            ax.flat[n].tick_params(axis="y",direction="inout")
            ax.flat[n].tick_params(axis="x",direction="inout")
            
            if j==0:
                n +=1   
            
        ax.flat[0].set_xticklabels([])
        ax.flat[1].set_xticklabels([])   
        if snap ==78:
            ax.flat[0].set_yticks([-11,-10,-9,-8])
            ax.flat[1].set_yticks([-11,-10,-9,-8])
            ax.flat[2].set_yticks([-11,-10,-9,-8])
            ax.flat[3].set_yticks([-11,-10,-9,-8])   
            ax.flat[0].set_xticks([10,11,12])   
            ax.flat[1].set_xticks([10,11,12])   
            ax.flat[2].set_xticks([10,11,12])   
            ax.flat[3].set_xticks([10,11,12])   
            ax.flat[0].set_xticklabels([])   
            ax.flat[1].set_xticklabels([])
            ax.flat[1].set_yticklabels([])      
            ax.flat[3].set_yticklabels([])   

        else:
            ax.flat[0].set_yticks([-13,-12,-11,-10,-9])
            ax.flat[1].set_yticks([-13,-12,-11,-10,-9])
            ax.flat[2].set_yticks([-13,-12,-11,-10,-9])
            ax.flat[3].set_yticks([-13,-12,-11,-10,-9])
            ax.flat[0].set_xticks([10,11,12])   
            ax.flat[1].set_xticks([10,11,12])   
            ax.flat[2].set_xticks([10,11,12])   
            ax.flat[3].set_xticks([10,11,12])   
            ax.flat[0].set_xticklabels([])   
            ax.flat[1].set_xticklabels([])
            ax.flat[1].set_yticklabels([])      
            ax.flat[3].set_yticklabels([])   

        
        ax_up0 = ax.flat[0].twiny()
        ax_up0.set_xlim(ax.flat[0].get_xlim())
        ax_up0.set_xticklabels([])

        ax_up1 = ax.flat[1].twiny()
        ax_up1.set_xlim(ax.flat[1].get_xlim())
        ax_up1.set_xticklabels([])
        
        ax.flat[2].set_xlabel('$\mathrm{log} \, (M_{\star} \,/ \,\mathrm{M_{\odot}})$')
        ax.flat[3].set_xlabel('$\mathrm{log} \, (M_{\star} \,/ \,\mathrm{M_{\odot}})$')
        ax.flat[0].set_ylabel('$\mathrm{log \, (sSFR \,/\, yr^{-1}}$)')
        ax.flat[1].set_yticklabels([])
        ax.flat[2].set_ylabel('$\mathrm{log \, (sSFR \,/\, yr^{-1}}$)')
        ax.flat[3].set_yticklabels([])
        
        cbar_ax = fig.colorbar(plot1, ax=ax.ravel().tolist(), shrink =0.5,location="bottom", pad=0.09, fraction =0.07)
        cbar_ax.set_label(clabel)

        plt.show()

        fig.savefig(f'sSFR_fedd_z{z_round}.pdf',bbox_inches='tight', dpi = 1200)


fb_fols  = ['noagn','nojet','nox','7jk']            
fb_types = ['No-AGN','No-jet','No-X-ray', 'Simba-50']
      
snaps = [151] 
sSFR_mStar_subplots(model, snaps,size, fb_fols, fb_types)        

snaps = [78] 
sSFR_mStar_subplots(model, snaps,size, fb_fols, fb_types) 
