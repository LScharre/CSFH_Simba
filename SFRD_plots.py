import numpy as np
import matplotlib.pyplot as plt
import caesar
from astropy.cosmology import WMAP9 as cosmo
import re
from matplotlib.offsetbox import AnchoredText
import os

#%%
fb_fol = 's50'
model = "m50n512"
data_cat =  #folder with sub folders 'nofb', 'noagn','nojet','nox','7jk' with respective Simba galaxy catalogues

size = 50
snaps = [151, 142,125, 104,90, 78,71, 62, 50,42, 36,30,26,22]


# my local folders 
fb_types = ['x-ray','jets','AGN winds',  'stellar', 'no fb']
fb_fols = ['7jk', 'nox','nojet',  'noagn', 'nofb']
colors = [ 'seagreen' ,'tomato', 'dodgerblue','darkmagenta','orange']


#%% compute SFRD 

def SFRD(model, fb_fol, snap, size):

    infile = f'{data_cat}{fb_fol}/{model}_{snap:03d}.hdf5'

    sim = caesar.load(infile)

    h = sim.simulation.hubble_constant
    z = sim.simulation.redshift
    if z > 1.99:
        z_round = round(z)
    elif 2 > z > 0:
        z_round = round(z, 2)

    SFR = np.array([i.sfr for i in sim.halos])

    SFR_sum = np.sum(SFR)

    Vol_Mpc = (size/h)**3
    SFR_density = SFR_sum/Vol_Mpc
    age = cosmo.age(z).value
    return SFR_density, z, age


#%% plot SFRD

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

def SFRD_plot(fb_fols, fb_types, colors):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    
    for i in range(len(fb_fols)):
        fb_type = fb_types[i]
        fb_fol = fb_fols[i]
        color = colors[i]
        SFRDs = []
        redshifts = []
        ages = []
    
        for j in range(len(snaps)):
            snap = snaps[j]

            SFR_density, z, age = SFRD(model, fb_fol, snap, size)
            SFRDs.append(SFR_density)
            redshifts.append(z)
            ages.append(age)
        
        ax1.plot(ages, np.log10(SFRDs), label=fb_type, color=color, lw=2)


        fig.set_size_inches(8,6)
        ax1.set_ylim(-2.65, -0.35)  
        
    include_MD = False
    if include_MD == True:
        filein = open(
            "md14_data.txt", "r")
        
        z_obs = []
        logSFRD = []
        sighi = []
        siglo = []
        
        for line in filein.readlines():  # iterates through the lines in the file
            # check if the current line
            # starts with "#"
            if line.startswith("#"):
                continue
            else:
                tokens = re.split(' +', line)
        
                # appends data in their respective lists
                z_obs.append(float(tokens[0]))
                logSFRD.append(float(tokens[1]))
        
                sighi.append(float(tokens[2]))  # appends array to position list
                siglo.append(abs(float(tokens[3])))
        filein.close()
        obs_err = np.array(tuple(zip(siglo, sighi)))
        obs_err = [siglo, sighi]
        imf_factor = 1.8
        logSFRD = logSFRD - np.log10(imf_factor)
        
        ax1.errorbar(cosmo.age(z_obs).value, logSFRD, obs_err,
                     label='Maudau & Dickinson (2014)', fmt='o', color='k', lw =2)
    
        ax1.set_ylabel('$\mathrm{log\, (\, SFRD\, /\, M_{\odot} \, yr^{-1} \, c M p c^{-3}\, )}$')
        
    include_AM = False
    if include_AM == True:
        #Analytic Model
        folder = "AnalyticModel/param_exploration"
        AM_colors = ["indigo","mediumpurple", "green"]
        i = 0
        for file in sorted(os.listdir(folder)):
          
            if file.startswith('CSFRD'):
    
                Data    = np.loadtxt(folder + '/' + file, comments='#', skiprows=1).T
                z = Data[0][Data[0]>0]
                rho = Data[1][Data[0]>0]
                
                temp = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", file)
                res = list(map(float, temp))
                print(res)
                if ((res[0] == 1.9) & (res[1] == 3.87))  or ((res[0] == 2.41) & (res[1] == 1.46)):
                    print()
                    ax1.plot(cosmo.age(z).value,np.log10(rho), label ="Sorini & Peacock (2021)", linestyle = "--",color=AM_colors[i], lw=2)
                    i+=1
            
    redshift_ticks = [0, 1, 2, 4, 6]
    age_ticks = cosmo.age(redshift_ticks).value
    ax1.set_xlabel('Cosmic Time [Gyr]')
    
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(age_ticks)
    ax2.set_xlabel('$z$')
    ax2.set_xticklabels(redshift_ticks)
    plt.rc('legend', fontsize=13)    # legend fontsize
    ax1.legend(loc='lower center',frameon=False)
    
    plt.savefig("SFRD_comparison.pdf", bbox_inches='tight', dpi=1200)


snaps = [151, 142,125, 104,90, 78,71, 62, 50,42, 36,30,26,22]

fb_fols  = ['nofb','noagn','nojet','nox','7jk']   
fb_types = ['No-feedback', 'No-AGN','No-jet','No-X-ray', 'Simba-50']
colors = ['orange','darkmagenta','dodgerblue', 'tomato',  'seagreen' ]

SFRD_plot(fb_fols, fb_types, colors)

#%% compute SFRD in SFR bins, called in SFRD_binned_plots
    
start = -1
stop = 4
step = 1
SFR_bins = np.arange(start,stop,step)

def SFRD_binned(model, fb_fol, snap, size, SFR_bins):
    infile = f'{data_cat}{fb_fol}/{model}_{snap:03d}.hdf5'
    sim = caesar.load(infile)

    h = sim.simulation.hubble_constant
    z = sim.simulation.redshift
    if z > 1.99:
        z_round = round(z)
    elif 2 > z > 0:
        z_round = round(z, 2)

    mH_tot = np.array([i.halo.masses['total']
                      for i in sim.galaxies if i.central == 1])
    mH_tot_lg = np.log10(mH_tot)

    SFR = np.array([i.halo.sfr for i in sim.galaxies if i.central == 1])
    
    SFR_nonzero = SFR[SFR>0]
    SFR_lg = np.log10(SFR_nonzero)

    Vol_Mpc = (size/h)**3

    SFRD_SFRbinned = []
    
    for bin_i in SFR_bins:
            masked_SFR_lg = np.ma.masked_inside(SFR_lg, bin_i-step, bin_i+step)
            SFR_mask = masked_SFR_lg.mask
            SFR_SFR_masked = SFR_nonzero[SFR_mask]
            SFRD_SFRbinned.append(np.sum(SFR_SFR_masked)/Vol_Mpc)
   
    SFRD_all = np.sum(SFR)/Vol_Mpc
    age = cosmo.age(z).value
    return SFRD_SFRbinned, SFRD_all, z, age


#%% SFRD convergence test
import cmasher as cmr   
def SFRD_runs_plot(models, sizes,snaps,labels):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    
    for i in range(len(models)):
        label = labels[i]
        model = models[i]
        color = colors[i]
        size = sizes[i]
        SFRDs = []
        redshifts = []
        ages = []
    
        for j in range(len(snaps)):
            snap = snaps[j]
            SFR_density, z, age = SFRD(model, fb_fol, snap, size)
            SFRDs.append(SFR_density)
            redshifts.append(z)
            ages.append(age)
            ax1.plot(ages, np.log10(SFRDs), label=label, color=color)
            

    ax1.set_ylabel('log SFRD ($M_{\odot} y r^{-1} M p c^{-3}$)')
    ax1.set_ylim(-2.65, -0.4)  
    filename = 'SFRD_convergence.pdf'

    redshift_ticks = [0, 1, 2, 4, 6]
    age_ticks = cosmo.age(redshift_ticks).value
    ax1.set_xlabel('Cosmic Time [Gyr]')
    
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(age_ticks)
    ax2.set_xlabel('Redshift')
    ax2.set_xticklabels(redshift_ticks)
    
    ax1.legend()
    plt.savefig(filename, bbox_inches='tight')


# models on server
models = ['m50n512','m25n512','m25n256', 'm100n1024']
labels = ['Simba-50','Simba-25 High-res.','Simba-25', 'Simba-100']
sizes = [50,25,25,100]
snaps = [151, 142,125, 104,90, 78,71, 62, 50,42, 36,30,26,22]


#SFRD_runs_plot(models,sizes,snaps,labels)

#%% only plot SFRD in SFR bins for AGN winds, jets and x-ray runs 

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


def SFRD_SFR_subplots(model,size,SFR_bins):
    
    fig1, ax1 = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(wspace=0.2, hspace=0.1)
    fig1.set_size_inches(11,9)

    # make different plots for different fb types
    fb_fols  = ['nojet','nox','7jk']  
    fb_types= ['AGN winds', 'jets','x-ray']  
    #fb_fols  = ['nofb','noagn','nojet','nox','7jk']   
    fb_types = ['No-jet','No-X-ray', 'Simba-50']
    print(fb_fols)
    for i in range(len(fb_fols)):

        if i !=3:
            snaps = [151, 142,125, 104,90, 78,71, 62, 50,42, 36,30,26,22]
            fb_type = fb_types[i]
            fb_fol = fb_fols[i]
            #color = colors[i]
            SFRDs = []
            redshifts = []
            ages = []
            SFRDs_SFRbinned =[]
            
            
            for j in range(len(snaps)):
                snap = snaps[j]
    
                SFRD_SFRbinned, SFRD_all, z, age = SFRD_binned(model, fb_fol, snap, size, SFR_bins)
                
                # [[massbinned_SFRD_z1],..., [massbinned_SFRD_zn]]
                
                SFRDs_SFRbinned.append(SFRD_SFRbinned)
                SFRDs.append(SFRD_all)
                
                redshifts.append(z)
                ages.append(age)
            
            # need to transpose, [[z_SFRD_m1],..., [z_SFRD_mn]], for each mass bin i now have a z evolution of SFRD

            SFRDs_SFRbinned_z = np.array(SFRDs_SFRbinned).T
            
            
            colormap = plt.cm.gist_ncar
            #cmap = cmr.get_sub_cmap('coolwarm_r', 0, 1)
            ax1.flat[i].set_prop_cycle(plt.cycler('color', plt.cm.coolwarm_r(np.linspace(0, 1, 4))))
    
            SFR_lines = []
            #labels = []
            for ii in range(len(SFRDs_SFRbinned_z)):
                if fb_type == "No-jet":
                    if ii==4:
                        line = ax1.flat[i].plot(ages, np.log10(SFRDs_SFRbinned_z[ii]), label=SFR_bins[ii], lw=2, color ="midnightblue")
                    else:
                        line = ax1.flat[i].plot(ages, np.log10(SFRDs_SFRbinned_z[ii]), label=SFR_bins[ii], lw=2)
                else:
                    if ii==4:
                        line = ax1.flat[i].plot(ages, np.log10(SFRDs_SFRbinned_z[ii]), lw=2, color ="midnightblue")
                    else:
                        line = ax1.flat[i].plot(ages, np.log10(SFRDs_SFRbinned_z[ii]), lw=2)

                
                SFR_lines.append(line)
                #labels.label()
                
            
            ax1.flat[i].plot(ages, np.log10(SFRDs), label=fb_type, color ='k', lw=2.5, ls="--")
            
            if fb_type == "No-jet":
                ax1.flat[1].plot(ages, np.log10(SFRDs), label='No-jet', color ='grey', lw=2.5, ls="--")
                ax1.flat[2].plot(ages, np.log10(SFRDs), label='No-jet', color ='lightgrey', lw=2.5, ls="--")
                
            if fb_type == "No-X-ray":
                
                ax1.flat[2].plot(ages, np.log10(SFRDs), label='No-X-ray', color ='grey', lw=2.5, ls="--")
                
    
            #ax1.set_ylabel('log SFRD ($M_{\odot} y r^{-1} M p c^{-3}$)')
            ax1.flat[i].set_ylim(-4.5 ,0)  
            at = AnchoredText('%s' %(fb_type), prop=dict(size=14), frameon=True, loc='upper right')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax1.flat[i].add_artist(at)
            
            #filename = 'SFRD_SFRbinned_%s.png'
            #ax1.flat[3].plot(ages, np.log10(SFRDs), label='total', color ='w')
            
            ax1.flat[i].tick_params(right=True, top=True)   
            ax1.flat[i].tick_params(axis="y",direction="inout")
            ax1.flat[i].tick_params(axis="x",direction="inout")
    

            redshift_ticks = [0, 1, 2, 4, 6]
            age_ticks = cosmo.age(redshift_ticks).value
        
            ax_up0 = ax1.flat[i].twiny()
            ax_up0.set_xlim(ax1.flat[0].get_xlim())
            ax_up0.set_xticks(age_ticks)
            if i <2:
                ax_up0.set_xlabel('$z$')
                ax_up0.set_xticklabels(redshift_ticks)
            else:
                ax_up0.set_xticklabels([])

    
    ax1.flat[0].legend(ncol=3, title = '$\mathrm{log( \,SFR  \,/  \,M_{\odot} \, yr^{-1})}$', fontsize =10,frameon =False)
    ax1.flat[1].legend(loc= "lower right", fontsize=10, frameon =False)
    ax1.flat[2].legend(loc= "lower right", fontsize=10, frameon =False)
    
    ax1.flat[0].set_xlim(0.2,14)
    ax1.flat[1].set_xlim(0.2,14)
    ax1.flat[2].set_xlim(0.2,14)
    ax1.flat[3].set_xlim(0.2,14)
    ax1.flat[0].set_xticklabels([])
    ax1.flat[1].set_xticklabels([])
   
    ax1.flat[2].set_xlabel('Cosmic Time [Gyr]')
    ax1.flat[1].set_ylabel('$\mathrm{log\, (\, SFRD\, /\, M_{\odot} \, yr^{-1} \, c M p c^{-3}\, )}$')
    ax1.flat[0].set_ylabel('$\mathrm{log\, (\, SFRD\, /\, M_{\odot} \, yr^{-1} \, c M p c^{-3}\, )}$')
    ax1.flat[2].set_ylabel('$\mathrm{log\, (\, SFRD\, /\, M_{\odot} \, yr^{-1} \, c M p c^{-3}\, )}$')

    ax1.flat[0].set_yticks([-4,-3,-2,-1,0])
    ax1.flat[1].set_yticks([-4,-3,-2,-1,0])
    ax1.flat[2].set_yticks([-4,-3,-2,-1,0])

    filename_SFR = 'SFRD_SFRbinned.pdf'
    
    fb_fols  = ['nox','7jk']  
    fb_types= [ 'jets','x-ray']  
    snaps = [151, 142,125, 104,90, 78,71, 62, 50,42]
    snaps =[51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,78,90, 104, 125, 142, 151]

    ax2 = ax1.flat[3]
    SFRDs_high_x =[]
    SFRDs_high_jets =[]
    SFRDs_x= []
    SFRDs_jets = []
    for i in range(len(fb_fols)):
        fb_type = fb_types[i]
        fb_fol = fb_fols[i]
        
        redshifts = []        
        ages = []        
        for j in range(len(snaps)):
            snap = snaps[j]
            infile = f'{data_cat}{fb_fol}/{model}_{snap:03d}.hdf5'
            print(infile)
            sim = caesar.load(infile)
        
            h = sim.simulation.hubble_constant
            z = sim.simulation.redshift
            if z > 1.99:
                z_round = round(z)
            elif 2 > z > 0:
                z_round = round(z, 2)
            print(z)
            print(' ')
            print(' ')
            
        
            SFR = np.array([i.halo.sfr for i in sim.galaxies if i.central == 1])
            mH_stellar = np.array([i.masses['stellar'] for i in sim.galaxies if i.central==1])
            SFR_bin = 1
            SFR_high = SFR[SFR>10**SFR_bin]
            sSFR = SFR[SFR>10**SFR_bin]/mH_stellar[SFR>10**SFR_bin]
            #print(sSFR.min(),sSFR.max())
            
            Vol_Mpc = (size/h)**3
            
            
            SFR_sum = np.sum(SFR)
            SFR_sum_high = np.sum(SFR_high)
            
            SFRD_all = SFR_sum/Vol_Mpc
            SFRD_high = SFR_sum_high/Vol_Mpc

            if fb_fol == '7jk': 
                SFRDs_high_x.append(SFRD_high)
                SFRDs_x.append(SFRD_all)
                
            elif fb_fol == 'nox': 
                SFRDs_high_jets.append(SFRD_high)
                SFRDs_jets.append(SFRD_all)
                
                
            redshifts.append(z)
            age = cosmo.age(z).value
            ages.append(age)

    ax1.flat[3].plot(ages, np.array(SFRDs_x)/np.array(SFRDs_jets), label='All SFR', color='k', lw=2)
    ax1.flat[3].plot(ages,  np.array(SFRDs_high_x)/np.array(SFRDs_high_jets), label='$\mathrm{log( \,SFR  \,/  \,M_{\odot} \, yr^{-1}) \geq 1}$', color='cornflowerblue', lw=2)
    ax1.flat[3].set_yticks([0.2,0.4,0.6,0.8,1])
    redshift_ticks = [0, 1, 2, 4, 6]
    age_ticks = cosmo.age(redshift_ticks).value

    ax_up0 = ax1.flat[3].twiny()
    ax_up0.set_xlim(ax1.flat[0].get_xlim())
    ax_up0.set_xticks(age_ticks)

    ax_up0.set_xticklabels([])
    
    ax1.flat[3].set_xlabel('Cosmic Time [Gyr]')
    ax1.flat[3].set_ylabel('SFRD (Simba-50 / No-X-ray)')
    ax1.flat[3].legend(loc= "lower left", fontsize=10, frameon =False)

    fig1.savefig(filename_SFR, bbox_inches='tight', dpi=1200)
    
SFRD_SFR_subplots(model,size,SFR_bins)      
    

