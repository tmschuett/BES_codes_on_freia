from imports import *
from BESClass import *
from functions_BES import *
from plotting_functions_BES_velocimetry_results import *
from plotting_functions_BES_velocimetry_checks import *

# collection of plotting functions for BES things

def plot_bes_locs(BESdata, timepoint):
    # plot the location of the BES view array with flux surfaces
    shot = BESdata.shot_number
    apdpos = BESdata.apdpos
    
    equilib = eq.equilibrium(device='MAST', shot=shot, time=timepoint)
    
    figure, axes = plt.subplots(1, 1, figsize=(7.5, 5.0))
    axes.contour(equilib.R, equilib.Z, np.sqrt(equilib.psiN), 
                 np.linspace(0, 1, 21), colors='k')
    axes.plot(apdpos[:, 0], apdpos[:, 1], 'or')
    axes.set_aspect('equal', adjustable='datalim')
    axes.set_xlabel('radius R [m]', fontsize=18)
    axes.set_ylabel('height above midplane z [m]', fontsize=18)
    axes.set_xlim([np.min(apdpos[:, 0]) - 0.035, np.max(apdpos[:, 0]) + 
        0.035])
    axes.set_ylim([-0.05, 0.05])
    axes.tick_params(axis='x', labelsize=16)
    axes.tick_params(axis='y', labelsize=16)
    axes.set_title(str(shot) + ', ' + str('{:<06}'.format(
        round(timepoint, 4))) + 's', fontsize=22)
    plt.savefig('shot' + str(shot) + '_BES_locs_' + 
                str(int(1000*round(timepoint, 3))) +'ms.png', 
                format='png', transparent=True)
    plt.close()

def plot_bes_fluct_spectrum(BESdata, column, timeslice, fname):
    # plot spectra of BES fluctuation data in a column for specified timeslice
    colours = ['k', 'r', 'b', 'g']
    
    bes_time = BESdata.time.cut
    fluct_data = BESdata.data.fluct
    nrows = BESdata.nrows
    ncols = BESdata.ncols
    shot = BESdata.shot_number
    
    idx1 = (np.abs(bes_time - timeslice[0])).argmin()
    idx2 = (np.abs(bes_time - timeslice[1])).argmin()
    plt.figure(figsize=(7.5, 5.5))
    for n in range(nrows):
        ch = ncols * n + column
        ps_bes = np.abs(np.fft.fft(fluct_data[ch][idx1:idx2])) ** 2
        time_step_bes = bes_time[idx1+1] - bes_time[idx1]
        freqs_bes = np.fft.fftfreq(fluct_data[ch][idx1:idx2].size, 
                                   time_step_bes)
        idx_bes = np.argsort(freqs_bes)
        plt.plot((1.0e-3 * freqs_bes[idx_bes]), (ps_bes[idx_bes]), 
                 color=(colours[n]), label=('ch' + str(ch)))

    plt.legend(loc='upper right', fontsize=18)
    plt.xlabel('frequency [kHz]', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xscale('log')
    plt.xlim([10, 1.0e3])
    plt.yscale('log')
    plt.title(('shot' + str(shot) + ' col ' + str(column) + ', times ' + 
               str([round(bes_time[idx1], 7), round(bes_time[idx2], 7)])), 
        fontsize=32)
    plt.savefig(('shot' + str(shot) + '_col' + str(column) + '_spectrum_' + 
                 str(fname) + '.png'), format='png', transparent=True)
    plt.close()

def plot_bes_fluctuations(BESdata, timeslice, channels, fname, threshold=None):
    # plot BES fluctuations for specified channels
    
    shot = BESdata.shot_number
    bes_time = BESdata.time.cut
    fluct_data = BESdata.data.fluct
    
    idx1 = (np.abs(bes_time - timeslice[0])).argmin()
    idx2 = (np.abs(bes_time - timeslice[1])).argmin()
    
    num_channels = len(channels)
    figure, axes = plt.subplots(num_channels, 1, sharex=True, 
                                figsize=(15, 3*num_channels))
    axes[0].set_title((str(shot) + ' BES fluctuations'), fontsize=32)
    for i in range(num_channels):
        ch = channels[i]
        axes[i].plot(bes_time[idx1:idx2], fluct_data[ch, idx1:idx2], 'k', 
            linewidth=0.8)
        axes[i].tick_params(axis='y', labelsize=22)
        axes[i].set_ylabel(('ch ' + str(ch)), fontsize=24)
        if threshold is not None:
            axes[i].axvline(threshold, color='r')
    axes[num_channels].tick_params(axis='x', labelsize=24)
    axes[num_channels].set_xlabel('time [s]', fontsize=26)
    axes[num_channels].set_xlim(timeslice)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(('shot' + str(shot) + '_BES_fluctuations_' + str(fname) + 
                 '.png'), format='png', transparent=True)
    plt.close()

def plot_bes_fluct_spectrogram(BESdata, channels, timeslice, n, fname, 
        threshold=None, freq_lims=[0.0, 200.0]):
    # plot the BES fluctuation data spectrogram for one or more channels
    shot = BESdata.shot_number
    bes_time = BESdata.time.cut
    fluct_data = BESdata.data.fluct
    
    idx1 = (np.abs(bes_time - timeslice[0])).argmin()
    idx2 = (np.abs(bes_time - timeslice[1])).argmin()
    
    freq, times, Sxx = sig.spectrogram(fluct_data[:,idx1:idx2], fs=f_samp, 
                                       nperseg=(2 ** n), scaling='spectrum')
    
    if isinstance(channels, list):
        num_chans = len(channels)
        
        new_lim = int(14 * (2 ** (n - 7)))
        
        figure, axes = plt.subplots(num_channels, 1, sharex=True, 
                                    figsize=(15, 3 * num_channels))
        axes[0].set_title(str(shot) + ' BES spectrograms [kHz], n=' + 
            str(int(2 ** n)), fontsize=32)
        for i in range(num_channels):
            ch = channels[i]
            axes[i].contourf(times + bes_time[idx1], 0.001 * freq[:new_lim], 
                Sxx[ch][:new_lim,:], 16, cmap=plt.get_cmap('gnuplot2'), 
                norm=(colors.LogNorm()))
            axes[i].set_ylim(freq_lims)
            axes[i].tick_params(axis='y', labelsize=22)
            axes[i].set_ylabel(('ch ' + str(ch)), fontsize=24)
            if threshold is not None:
                axes[i].axvline(threshold, color='r')
        axes[num_channels-1].tick_params(axis='x', labelsize=24)
        axes[num_channels-1].set_xlabel('time [s]', fontsize=26)
        axes[num_channels-1].set_xlim(timeslice)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig('shot' + str(shot) + '_spectrogram_'+fname+'.png', 
                    format='png', transparent=True)
        plt.close()
    else:
        end_lev = int(np.ceil(np.log10(np.max(Sxx[:][:int(
                np.shape(freq)[0]/5), :]))))
        levs = np.logspace(end_lev - 7, end_lev, 
                           num=16)
        
        plt.figure(figsize=(10, 10))
        plt.title(str(shot) + ', BES ch=' + str(channels) + ', n=' + 
                  str(int(2 ** n)), fontsize=28)
        plt.contourf(times + bes_time[idx1], 0.001 * freq, Sxx[channels], 
                     levs, cmap=plt.get_cmap('gnuplot2'), 
                     norm=(colors.LogNorm()))
        plt.ylim(freq_lims)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.ylabel('frequency [kHz]', fontsize=26)
        if threshold is not None:
            plt.axvline(threshold, color='r')
        plt.xlabel('time [s]', fontsize=26)
        plt.xlim(timeslice)
        plt.savefig('shot' + str(shot) + '_ch' + str(channels) + 
                    '_BES_spectrogram_n' + str(int(2 ** n)) + '_' + fname + 
                    '.png', format='png', transparent=True)
        plt.close()

def plot_with_flux_surfaces(
        BESdata, equilib_R, equilib_Z, equilib_psi, quantity, title, 
        timepoint, fname, max_val, cmap=plt.cm.RdBu_r, dalpha=None, 
        num_contours=11, save_format='png'):
    # plot the data array with flux surfaces and BES channel locations 
    # if plotting with Dalpha, set dalpha=[dalpha_time, dalpha_data]
    apdpos = BESdata.apdpos
    nrows = BESdata.nrows
    ncols = BESdata.ncols
    if dalpha is not None:
        figure, axes = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8.5, 8.5))
        ax0 = axes[0]
        font_size = 18
        r_lim = 0.035
        z_lim = 0.05
    else:
        figure, axes = plt.subplots(1, 1, figsize=(8.8, 5.8))
        ax0 = axes
        font_size = 28
        r_lim = 0.005
        z_lim = 0.04
    ax0.contour(equilib_R, equilib_Z, equilib_psi, 
                np.linspace(0, 1, num_contours), colors='k')
    ax0.plot(apdpos[:, 0], apdpos[:, 1], 'or')
    cs = ax0.contourf(apdpos[:, 0].reshape(nrows, ncols), 
                      apdpos[:, 1].reshape(nrows, ncols),
                      np.asarray(quantity).reshape(nrows, ncols), 12, 
                      vmin=-max_val, vmax=max_val, cmap=cmap)
    ax0.set_aspect('equal', adjustable='datalim')
    ax0.set_xlabel('radius R [m]', fontsize=font_size)
    ax0.set_ylabel('height above midplane z [m]', fontsize=font_size)
    ax0.set_xlim([np.min(apdpos[:, 0]) - r_lim, 
                  np.max(apdpos[:, 0]) + r_lim])
    ax0.set_ylim([-z_lim, z_lim])
    ax0.tick_params(axis='x', labelsize=font_size)
    ax0.tick_params(axis='y', labelsize=font_size)
    ax0.set_title(title + str('{:<09}'.format(round(timepoint, 7))), 
        fontsize=font_size)
    if dalpha is not None:
        dalpha_time = dalpha[0]
        dalpha_data = dalpha[1]
        axes[1].plot(dalpha_time, dalpha_data, 'b')
        axes[1].axvline(timepoint, color='r')
        axes[1].set_xlabel('time [s]', fontsize=font_size)
        axes[1].set_ylabel('D alpha', fontsize=font_size)
        axes[1].tick_params(axis='x', labelsize=font_size)
        axes[1].tick_params(axis='y', labelsize=font_size)
    else:
        cbar = figure.colorbar(cs)
        cbar.ax.tick_params(labelsize=font_size, width=2)
        plt.tight_layout()
    plt.savefig(fname, format=save_format, transparent=True)
    plt.close()
