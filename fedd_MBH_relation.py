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


#%%
def fedd_mBH_subplots(model, snaps,size, fb_fols, fb_types, z_data ="sSFR") :
        SMALL_SIZE = 8
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 12
        plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=12)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.rcParams['legend.title_fontsize'] = 12
        fig, ax = plt.subplots(nrows=4, ncols=2)
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.set_size_inches(10,13)
      
        n =0
        for j in range(len(snaps)):

            snap = snaps[j]
            for i in range(len(fb_types)):
                fb_type = fb_types[i]
                fb_fol = fb_fols[i]
                
                infile = f'{data_cat}{fb_fol}/{model}_%03d.hdf5' %(snap)
                sim = caesar.load(infile)
                
                h = sim.simulation.hubble_constant    
                z = sim.simulation.redshift
                z_round =int(round(z,0))

                mH_tot = np.array([i.halo.masses['total'] for i in sim.galaxies if i.central==1])
                mH_stellar = np.array([i.masses['stellar'] for i in sim.galaxies if i.central==1])
                mH_BH = np.array([i.masses['bh'] for i in sim.galaxies if i.central==1])
                
                SFR = np.array([i.halo.sfr for i in sim.galaxies if i.central==1])
                bh_fedd= np.array([i.halo.bh_fedd for i in sim.galaxies if i.central==1])  
                mH_gas = np.array([i.masses['gas'] for i in sim.galaxies if i.central==1])
                
                T =np.array([i.temperatures['mass_weighted'] for i in sim.galaxies if i.central==1])

                gas_frac = mH_gas/mH_stellar
                gas_th = 0.2
                
                mask = (mH_BH>0)  
                
                mH_tot = mH_tot[mask]
                mH_stellar = mH_stellar[mask]
                
                SFR =SFR[mask]
                bh_fedd= bh_fedd[mask]
                mH_gas = mH_gas[mask]
                mH_BH = mH_BH[mask]
                
                BHm_frac = np.log10(mH_BH/mH_stellar)
                gas_frac = mH_gas/mH_stellar
                sSFR = SFR/mH_stellar
                
                bh_fedd[bh_fedd == 0] = 1e-7
                gas_frac[gas_frac == 0] = 1e-7
                sSFR[sSFR == 0] = 1e-20

                alpha=1
                s_1 = 1
                s_2 = 1
                s_3 = 5
                s_4 = 30
                s_5 = 100
                vmin = 2e-12
                vmax = 1.6e-9
                size_array = [1,5,30,60]
                labels = ['$M_\mathrm{BH}/M_{\star} < 10^{-3}$', '$M_\mathrm{BH}/M_{\star} < 10^{-2.5}$', '$M_\mathrm{BH}/M_{\star} < 10^{-2}$', '$M_\mathrm{BH}/M_{\star} < 10^{-1}$']
                markers = ["o","o","o","o"]
                clabel = '$\mathrm{log \, (sSFR \,/\, yr^{-1}}$)'
                
                cmap = "jet_r"
                
                wind_mask = np.ma.masked_inside(mH_BH, 0, 10**7.5).mask | (bh_fedd>0.2)
                jet_mask = (mH_BH >  10**7.5) & (bh_fedd<0.2) & (gas_frac>0.2)
                xray_mask = (mH_BH >  10**7.5) & (bh_fedd<0.2) & (gas_frac<0.2)
                bh_fedd = np.log10(bh_fedd)

                # WINDS
                plot1 = ax.flat[n].scatter(np.log10(mH_BH[wind_mask & (BHm_frac < -4)]),        bh_fedd[wind_mask & (BHm_frac < -4)],         c= sSFR[wind_mask & (BHm_frac < -4)],           norm=mlp.colors.LogNorm(vmin,vmax), s=s_1,cmap=cmap, alpha =alpha)
                plot = ax.flat[n].scatter(np.log10(mH_BH[wind_mask & (BHm_frac < -3) & (BHm_frac > -4)]),  bh_fedd[wind_mask & (BHm_frac < -3) & (BHm_frac > -4)],  c= sSFR[wind_mask & (BHm_frac < -3) & (BHm_frac > -4)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_2,    cmap=cmap, alpha =alpha, edgecolor =None)
                plot = ax.flat[n].scatter(np.log10(mH_BH[wind_mask & (BHm_frac < -2.5) & (BHm_frac > -3)]),  bh_fedd[wind_mask & (BHm_frac < -2.5) & (BHm_frac > -3)],  c= sSFR[wind_mask & (BHm_frac < -2.5) & (BHm_frac > -3)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_3,    cmap=cmap, alpha =alpha, edgecolor =None)
                plot = ax.flat[n].scatter(np.log10(mH_BH[wind_mask & (BHm_frac < -2) & (BHm_frac > -2.5)]),  bh_fedd[wind_mask & (BHm_frac < -2) & (BHm_frac > -2.5)], c= sSFR[wind_mask & (BHm_frac < -2) & (BHm_frac > -2.5)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_4,    cmap=cmap, alpha =alpha, edgecolor =None)
                plot = ax.flat[n].scatter(np.log10(mH_BH[wind_mask & (BHm_frac < -1) & (BHm_frac > -2)]),  bh_fedd[wind_mask & (BHm_frac < -1) & (BHm_frac > -2)], c= sSFR[wind_mask & (BHm_frac < -1) & (BHm_frac > -2)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_5,    cmap=cmap, alpha =alpha, edgecolor =None)

                # JETS 
                ec = "k"
                lw=0.1
                plot = ax.flat[n].scatter(np.log10(mH_BH[jet_mask & (BHm_frac < -4)]),        bh_fedd[jet_mask & (BHm_frac < -4)],         c= sSFR[jet_mask & (BHm_frac < -4)],           norm=mlp.colors.LogNorm(vmin,vmax), s=s_1,cmap=cmap, marker = 's', alpha =alpha, edgecolor =ec,linewidths=lw)
                plot = ax.flat[n].scatter(np.log10(mH_BH[jet_mask & (BHm_frac < -3) & (BHm_frac > -4)]),  bh_fedd[jet_mask & (BHm_frac < -3) & (BHm_frac > -4)],  c= sSFR[jet_mask & (BHm_frac < -3) & (BHm_frac > -4)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_2,cmap=cmap, marker = 's', alpha =alpha, edgecolor =ec,linewidths=lw)
                plot = ax.flat[n].scatter(np.log10(mH_BH[jet_mask & (BHm_frac < -2.5) & (BHm_frac > -3)]),  bh_fedd[jet_mask & (BHm_frac < -2.5) & (BHm_frac > -3)],  c= sSFR[jet_mask & (BHm_frac < -2.5) & (BHm_frac > -3)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_3,cmap=cmap, marker = 's', alpha =alpha, edgecolor =ec,linewidths=lw)
                plot = ax.flat[n].scatter(np.log10(mH_BH[jet_mask & (BHm_frac < -2) & (BHm_frac > -2.5)]),  bh_fedd[jet_mask & (BHm_frac < -2) & (BHm_frac > -2.5)],  c= sSFR[jet_mask & (BHm_frac < -2) & (BHm_frac > -2.5)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_4,cmap=cmap, marker = 's', alpha =alpha, edgecolor =ec,linewidths=lw)
                plot = ax.flat[n].scatter(np.log10(mH_BH[jet_mask & (BHm_frac < -1) & (BHm_frac > -2)]),  bh_fedd[jet_mask & (BHm_frac < -1) & (BHm_frac > -2)], c= sSFR[jet_mask & (BHm_frac < -1) & (BHm_frac > -2)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_5,  cmap=cmap, marker = 's', alpha =alpha, edgecolor =ec,linewidths=lw)
                   
                # X-RAY 
                alpha=0.8
                
                plot = ax.flat[n].scatter(np.log10(mH_BH[xray_mask & (BHm_frac < -4)]),        bh_fedd[xray_mask & (BHm_frac < -4)],         c= sSFR[xray_mask & (BHm_frac < -4)],           norm=mlp.colors.LogNorm(vmin,vmax), s=s_1,cmap=cmap, marker = 'v', alpha =alpha, edgecolor =None)
                plot = ax.flat[n].scatter(np.log10(mH_BH[xray_mask & (BHm_frac < -3) & (BHm_frac > -4)]),  bh_fedd[xray_mask & (BHm_frac < -3) & (BHm_frac > -4)],  c= sSFR[xray_mask & (BHm_frac < -3) & (BHm_frac > -4)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_2,cmap=cmap, marker = 'v', alpha =alpha, edgecolor =None)
                plot = ax.flat[n].scatter(np.log10(mH_BH[xray_mask & (BHm_frac < -2.5) & (BHm_frac > -3)]),  bh_fedd[xray_mask & (BHm_frac < -2.5) & (BHm_frac > -3)], c= sSFR[xray_mask & (BHm_frac < -2.5) & (BHm_frac > -3)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_3,cmap=cmap, marker = 'v', alpha =alpha, edgecolor =None)
                plot = ax.flat[n].scatter(np.log10(mH_BH[xray_mask & (BHm_frac < -2) & (BHm_frac > -2.5)]),  bh_fedd[xray_mask & (BHm_frac < -2) & (BHm_frac > -2.5)],  c= sSFR[xray_mask & (BHm_frac < -2) & (BHm_frac > -2.5)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_4,cmap=cmap, marker = 'v', alpha =alpha, edgecolor =None)
                plot = ax.flat[n].scatter(np.log10(mH_BH[xray_mask & (BHm_frac < -1) & (BHm_frac > -2)]),  bh_fedd[xray_mask & (BHm_frac < -1) & (BHm_frac > -2)], c= sSFR[xray_mask & (BHm_frac < -1) & (BHm_frac > -2)],    norm=mlp.colors.LogNorm(vmin,vmax), s=s_5,cmap=cmap, marker = 'v', alpha =alpha, edgecolor =None)
                     
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

                ax.flat[n].set_ylim(5e-8,1e1)
                ax.flat[n].set_xlim(1e6,1e11)
                
                ax.flat[n].set_ylim(np.log10(5e-8),np.log10(0.5e1))
                ax.flat[n].set_xlim(np.log10(8e5),np.log10(6e10))
                
                ax.flat[n].tick_params(right=True, top=True)   
                ax.flat[n].tick_params(axis="y",direction="inout")
                ax.flat[n].tick_params(axis="x",direction="inout")

                at = AnchoredText('$z$ = %s' %z_round, prop=dict(size=12), frameon=True, loc='upper right')
                at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
                ax.flat[n].add_artist(at)
                
                if n%2 !=0:
                    ax.flat[n].set_yticklabels([])
                    
                else:
                    ax.flat[n].set_ylabel('$\mathrm{log} \, (f_\mathrm{edd})$')
                    
                if n<6:
                    ax.flat[n].set_xticklabels([])
                    if n<2:
                        ax_up1 = ax.flat[n].twiny()
                        ax_up1.set_xticklabels([])
                        ax_up1.set_xticks([])
                        ax_up1.set_xlabel(fb_types[i], fontweight='bold' )
                    
                else:
                    ax.flat[n].set_xlabel('$\mathrm{log} \, (M_\mathrm{BH}  \, /  \,\mathrm{M_{\odot}})$')

                n +=1

        cbar_ax = fig.colorbar(plot, ax=ax.ravel().tolist(), shrink =0.5)
        cbar_ax.set_label(clabel)

        plt.show()

        fig.savefig(f'MBH_fedd_{z_data}.pdf',bbox_inches='tight', dpi =1200)
            


fb_fols  = ['nox','7jk']  
fb_types = ['No-X-ray', 'Simba-50']

snaps = [51,78,104,151]              

fedd_mBH_subplots(model, snaps,size, fb_fols, fb_types, z_data ="sSFR") 




