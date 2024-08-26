#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:20:55 2021

@author: luciescharre
"""

import caesar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from scipy import stats
from matplotlib.offsetbox import AnchoredText
import h5py

#%%
def percentile16(y):
   return(np.percentile(y,16))

def percentile84(y):
   return(np.percentile(y,84))
    


#%% 
model = "m50n512"  
size=50   
data_cat = #folder with sub folders 'nofb', 'noagn','nojet','nox','7jk' with respective Simba galaxy catalogues

server_fb_fols = ['s50noagn','s50nojet','s50nox', 's50'] 
fb_titles = ['+ stellar','+ AGN winds','+ jets','+ x-ray'] 
fb_types = ['noagn','nojet','nox','7jk'] 

UM_files = ["smhm_a1.002312.dat", "smhm_a0.334060.dat"]

snaps = [151,78]

def moster(M,z):
    
    # at z = 0
    M10 = 11.590     # characteristic mass
    beta10 = 1.376    # slope indicating behaviour at low-mass end 
    gamma10 = 0.608   # slope indicating behaviour at high-mass end 
    N10 = 0.0351    # normalisation 
    
    M11 = 1.195    # characteristic mass
    beta11 = -0.826   # slope indicating behaviour at low-mass end 
    gamma11 = 0.329   # slope indicating behaviour at high-mass end 
    N11 = -0.03247    # normalisation 
    
    M1 = 10**(M10 + M11*z)
    N = N10 + N11*z
    beta = beta10 *(z+1)**beta11
    gamma = gamma10 *(z+1)**gamma11
    
    # stellar mass halo mass relation, multiplied ratio in equation 2 by M
    m = M * 2 * N *((M/M1)**(-beta)+(M/M1)**(gamma))**(-1) 
    
    # return stellar masses
    return m 


def SMH_ratio(model, size, fb_types,server_fb_fols, fb_titles, snaps):
    for i in range(len(snaps)):
        snap = snaps[i]
        count = 9
        fig, ax = plt.subplots(nrows=2, ncols=2)
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.set_size_inches(12,9)
       
        for j in range(len(fb_types)):
            fb_type = fb_types[j]
            fb_title = fb_titles[j]
            
            count -= 1

            infile = f'{data_cat}{fb_type}/{model}_{snap:03d}.hdf5'
            sim = caesar.load(infile)
            
            h = sim.simulation.hubble_constant    
            z = sim.simulation.redshift
            if z > 2:
                z_round =float(int(round(z)))
            elif 2 > z > 0:
                z_round =round(int(round(z)))

            
            #mH_tot, mH_stellar, SFR, SFR_100, bhmdot, bh_fedd,mH_BH = read_cat(sim,  centralOnly = True)
            central = np.array([i.central for i in sim.galaxies])
            
            mH_tot = np.array([i.halo.masses['total'] for i in sim.galaxies if i.central==1])
            #i.masses only instead to do only the mass of the central galaxy
            # do i.halo.masses to do total mass of halo that has a central galaxy
            mH_stellar_gal = np.array([i.masses['stellar'] for i in sim.galaxies if i.central==1])
            
            mH_stellar_halo = np.array([i.halo.masses['stellar'] for i in sim.galaxies if i.central==1])
            SFR = np.array([i.halo.sfr for i in sim.galaxies if i.central==1])
            mH_200 = np.array([i.halo.virial_quantities['m200c']
                              for i in sim.galaxies if i.central == 1])

            mH_stellar = mH_stellar_gal
            SFR = SFR
            mH_tot = mH_tot
            
                
            mH_ratio = np.log10(mH_stellar/mH_tot)
                
            sSFR = SFR[mH_stellar > 0]/mH_stellar[mH_stellar > 0]
            sSFRmin = min(sSFR)
            sSFRmax = max(sSFR)
            
            sSFRmin = 1e-11
            sSFRmax = 1e-8
            
            if z < 1:
                res = 40
                #sSFRmin = 5e-11
            
            elif 1 <= z <= 2: 
                res = 30
                #sSFRmin = 5e-11
                
            elif 2 < z < 3: 
                res = 30
                #sSFRmin = 1e-10
            else: 
                res = 20
            
            res = 41
            
            Nbins1 = res
            Nbins2 = res
            
            
            if snap == 151:
            # hist will be an array with size Nbins1 X Nbins2; 
            # each element contains the number of haloes falling in the corresponding 2D bin of Mhalo and Mstar.
                Nbins1 = np.linspace(10,14,res)
                Nbins2 = np.linspace(-3.5,-0.5,res)
                hist,xedges,yedges = np.histogram2d(np.log10(mH_tot[mH_stellar > 0]),np.log10(mH_stellar[mH_stellar > 0]/mH_tot[mH_stellar > 0]),bins=(Nbins1,Nbins2))
                whist,xedges,yedges = np.histogram2d(np.log10(mH_tot[mH_stellar > 0]),np.log10(mH_stellar[mH_stellar > 0]/mH_tot[mH_stellar > 0]),bins=(Nbins1, Nbins2),weights=sSFR)

            
            else: 
                # hist will be an array with size Nbins1 X Nbins2; 
                # each element contains the number of haloes falling in the corresponding 2D bin of Mhalo and Mstar.
                Nbins1 = np.linspace(10,14,res)
                Nbins2 = np.linspace(-3.5,-0.5,res)
                
                hist,xedges,yedges = np.histogram2d(np.log10(mH_tot[mH_stellar > 0]),np.log10(mH_stellar[mH_stellar > 0]/mH_tot[mH_stellar > 0]), bins=(Nbins1,Nbins2))
                whist,xedges,yedges = np.histogram2d(np.log10(mH_tot[mH_stellar > 0]),np.log10(mH_stellar[mH_stellar > 0]/mH_tot[mH_stellar > 0]),bins=(Nbins1,Nbins2),weights=sSFR)

            #2D array of size Nbins1 X Nbins2, where each element contains by construction the MEAN sSFR of the haloes in the corresponding 2D bin
            r = whist/hist

            extent = [10,14,-3.5,-0.5]

        
            im = ax.flat[j].imshow(r.T, norm=mlp.colors.LogNorm(vmin=sSFRmin, vmax=sSFRmax), extent = extent, interpolation=None,origin='lower',aspect='auto',cmap="jet_r")
            
            ax.flat[j].set_ylim(-3.5,-0.4)
            ax.flat[j].set_xlim(9.8,14.2)
            ax.flat[j].set_yticks([-3,-2,-1])
            
            ax.flat[j].tick_params(axis="y",direction="inout")
            ax.flat[j].tick_params(axis="x",direction="inout")

            Nbin = 20
            Nbins_med = np.linspace(10,14,Nbin)

            
            M_data, bin_edges, binnumber = stats.binned_statistic(np.log10(mH_tot[mH_stellar > 0]),np.log10(mH_stellar[mH_stellar > 0]/mH_tot[mH_stellar > 0]),statistic='median', bins=Nbins_med)
            M_upper= stats.binned_statistic(np.log10(mH_tot[mH_stellar > 0]),np.log10(mH_stellar[mH_stellar > 0]/mH_tot[mH_stellar > 0]),statistic=percentile84, bins=Nbins_med)[0]  
            M_lower = stats.binned_statistic(np.log10(mH_tot[mH_stellar > 0]),np.log10(mH_stellar[mH_stellar > 0]/mH_tot[mH_stellar > 0]),statistic=percentile16, bins=Nbins_med)[0]  
            
            # find the bincentres 
            bincen = np.zeros(len(bin_edges)-1)
            for k in range(len(bin_edges)-1):
                bincen[k] = 0.5*(bin_edges[k]+bin_edges[k+1])
   
            bincen_mask = (bincen>10.6) & (bincen<14)
            if z_round ==2:
                bincen_mask = (bincen>11.7) & (bincen<13.5)
                color ="r"
                
                ax.flat[j].errorbar(bincen[bincen_mask],M_data[bincen_mask],color=color, capsize =3, label = "Median")
                ax.flat[j].errorbar(bincen[bincen_mask],M_upper[bincen_mask],linestyle ="-",color="k", capsize =3, label = "16-84th percentile" )
                ax.flat[j].errorbar(bincen[bincen_mask],M_lower[bincen_mask],linestyle ="-",color="k", capsize =3)
                
                bincen_mask = (bincen>10.6) & (bincen<13.5)
                ax.flat[j].errorbar(bincen[bincen_mask],M_data[bincen_mask],color=color,linestyle ="--", capsize =3)
                ax.flat[j].errorbar(bincen[bincen_mask],M_upper[bincen_mask],linestyle ="--",color="k", capsize =3, label="resolution cut biasing")
                ax.flat[j].errorbar(bincen[bincen_mask],M_lower[bincen_mask],linestyle ="--",color="k", capsize =3)
            
            elif z_round ==0:
                color ="b"
                bincen_mask = (bincen>11.2) & (bincen<14)
                ax.flat[j].errorbar(bincen[bincen_mask],M_data[bincen_mask],color=color, capsize =3, label = "Median")
                ax.flat[j].errorbar(bincen[bincen_mask],M_upper[bincen_mask],linestyle ="-",color="k", capsize =3, label = "16-84th percentile" )
                ax.flat[j].errorbar(bincen[bincen_mask],M_lower[bincen_mask],linestyle ="-",color="k", capsize =3)
                
                bincen_mask = (bincen>10.6) & (bincen<14)
                ax.flat[j].errorbar(bincen[bincen_mask],M_data[bincen_mask],color=color,linestyle ="--", capsize =3)
                ax.flat[j].errorbar(bincen[bincen_mask],M_upper[bincen_mask],linestyle ="--",color="k", capsize =3, label="resolution cut biasing")
                ax.flat[j].errorbar(bincen[bincen_mask],M_lower[bincen_mask],linestyle ="--",color="k", capsize =3)

            
            M_data = np.log10((10**M_data)/10**bincen)

            M_data, bin_edges, binnumber = stats.binned_statistic(np.log10(mH_tot[mH_stellar > 0]),np.log10(mH_stellar[mH_stellar > 0]/mH_tot[mH_stellar > 0]),statistic='median', bins=Nbins_med)
            M_upper= stats.binned_statistic(np.log10(mH_tot[mH_stellar > 0]),np.log10(mH_stellar[mH_stellar > 0]/mH_tot[mH_stellar > 0]),statistic=percentile84, bins=Nbins_med)[0]  
            M_lower = stats.binned_statistic(np.log10(mH_tot[mH_stellar > 0]),np.log10(mH_stellar[mH_stellar > 0]/mH_tot[mH_stellar > 0]),statistic=percentile16, bins=Nbins_med)[0]  

            UM_include = False
            if UM_include == True: # to include UniverseMachine data
                UM_file = UM_files[i]
               
                UM_data = np.loadtxt("umachine-dr1/data/smhm/median_fits/%s" %(UM_file), comments='#').T
                UM_HM = UM_data[0]
                UM_R = UM_data[4]
                UM_R_plus = UM_data[5]
                UM_R_minus = UM_data[6]
                
                ax.flat[j].errorbar(UM_HM[UM_R<0],UM_R[UM_R<0],yerr = [UM_R_plus[UM_R<0],UM_R_minus[UM_R<0]], linestyle = "-",linewidth=1,color="purple", label = "Behroozi et al. (2019)")          
            
            if j == 0 or j==1:
                ax.flat[j].set_xticklabels([])
            
            if j == 1 or j == 3:    
                ax.flat[j].set_yticklabels([])

            at = AnchoredText('%s' %(fb_title), prop=dict(size=14), frameon=True, loc='upper right')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.flat[j].add_artist(at)

           
            xbins = 10**np.linspace(10,14,40)
        
        ax.flat[2].set_ylabel('$\mathrm{log}(\,M_{\star}\,/\,M_{\mathrm{H}}\,)$')
        ax.flat[0].set_ylabel('$\mathrm{log}(\,M_{\star}\,/\,M_{\mathrm{H}}\,)$')
        ax.flat[3].set_xlabel('$\mathrm{log}(\,M_{\mathrm{H}} \, /\,\mathrm{M_{\odot}}\,$)')
        ax.flat[2].set_xlabel('$\mathrm{log}(\,M_{\mathrm{H}}\, /\,\mathrm{M_{\odot}} \,$)')
        ax.flat[3].legend(loc = "upper left", fontsize=10)
        
        cbar_ax = fig.colorbar(im, ax=ax.ravel().tolist(), shrink =0.5)
        cbar_ax.set_label('sSFR [$\mathrm{yr^{-1}}$]')
        
        plt.gcf().text(0.78, 0.8, '$z = %s$' %(int(z_round)), fontsize=25)
        plt.savefig(f'SMR_SFR_{z_round}_med_p16-84.pdf', bbox_inches='tight',dpi=1200)
    
        plt.show()

fb_types  = ['noagn','nojet','nox','7jk']  
fb_titles = [ 'No-AGN','No-jet','No-X-ray', 'Simba-50']
SMH_ratio(model, size, fb_types,server_fb_fols, fb_titles, snaps)     


