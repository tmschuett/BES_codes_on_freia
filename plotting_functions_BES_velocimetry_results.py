from imports import *
from BESClass import *
from functions_BES import *

# collection of plotting functions for BES velocimetry results

def plot_velocity_one_channel(
        BESdata, velocity_array, correlation_array, time_points, timeslice, 
        num_slices, length, direction, label, row, column, title, fname, 
        time_lims=None, threshold=None, n=7, end_lev=-1, v_lims=None, 
        freq_lims=[0.0, 200.0], dalpha=None, bes_data=False):
    # plot velocity for one channel, optionally with Dalpha and/or spectrogram 
    # of BES data (set dalpha=[dalpha_time, dalpha_data])
    shot = BESdata.shot_number
    apdpos = BESdata.apdpos
    f_samp = BESdata.f_samp
    ncols = BESdata.ncols
    
    channel = ncols * row + column
    
    vels_ch, corrs_ch = get_plausible_velocities(
        row, column, velocity_array, correlation_array, apdpos, num_slices, 
        direction=direction)
    
    window = np.hanning(length + 1)
    velocity_line = convolve(vels_ch, window)
    
    if bes_data is True:
        if dalpha is not None:
            figure, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 15))
            vel_axis = axes[0]
            bes_axis = axes[1]
            dalpha_axis = axes[2]
            bottom_axis = axes[2]
        else:
            figure, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 15))
            vel_axis = axes[0]
            bes_axis = axes[1]
            bottom_axis = axes[1]
    elif dalpha is not None:
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 15))
        vel_axis = axes[0]
        da_axis = axes[1]
        bottom_axis = axes[1]
    else:
        figure, axes = plt.subplots(1, 1, sharex=True, figsize=(15, 15))
        vel_axis = axes
        bottom_axis = axes
    
    vel_axis.set_title(str(shot) + ' ch ' + str(channel) + ', ' + title, 
        fontsize=32)
    vel_axis.axhline(0.0, linestyle='--', color='grey')
    vel_axis.scatter(time_points, (1.0e-3 * np.asarray(vels_ch)), 
        c=(np.asarray(corrs_ch)), marker='.', cmap=plt.get_cmap('spring_r'))
    vel_axis.plot(time_points, 1.0e-3 * velocity_line, 'k', label=label)
    vel_axis.set_ylabel(direction + ' velocity [km/s]', fontsize=32)
    vel_axis.tick_params(axis='y', labelsize=32)
    vel_axis.legend(loc='upper right', fontsize=32)
    if threshold is not None:
        vel_axis.axvline(threshold, color='r', linestyle='--')
    try:
        vel_axis.set_ylim(v_lims)
    except:
        pass
    
    if bes_data is True:
        bes_time = BESdata.time.cut
        fluct_data = BESdata.data.fluct
        start_idx = np.abs(bes_time - time_lims[0]).argmin()
        end_idx = np.abs(bes_time - time_lims[1]).argmin()
        freq, times, Sxx = sig.spectrogram(fluct_data[:, start_idx:end_idx], 
                                           fs=f_samp, nperseg=2 ** n, 
                                           scaling='spectrum')
        end_lev = int(np.ceil(np.log10(np.max(Sxx[:][:int(
            np.shape(freq)[0]/5), :]))))
        levs = np.logspace(end_lev - 7, end_lev, num=16)
        new_lim = int(14 * (2 ** (n - 7)))
        bes_axis.contourf(times + bes_time[start_idx], 0.001 * freq[:new_lim], 
            Sxx[channel][:new_lim,:], levs, cmap=plt.get_cmap('gnuplot2'), 
            norm=(colors.LogNorm()))
        bes_axis.set_ylim(freq_lims)
        bes_axis.tick_params(axis='y', labelsize=32)
        bes_axis.set_ylabel('frequency [kHz]', fontsize=32)
        if threshold is not None:
            bes_axis.axvline(threshold, color='r', linestyle='--')
    
    if dalpha is not None:
        dalpha_time = dalpha[0]
        dalpha_data = dalpha[1]
        try:
            dalpha_idx1 = (np.abs(dalpha_time - time_lims[0])).argmin()
            dalpha_idx2 = (np.abs(dalpha_time - time_lims[1])).argmin()
        except:
            dalpha_idx1 = (np.abs(dalpha_time - timeslice[0])).argmin()
            dalpha_idx2 = (np.abs(dalpha_time - timeslice[1])).argmin()
        da_axis.plot(dalpha_time[dalpha_idx1:dalpha_idx2], 
            dalpha_data[dalpha_idx1:dalpha_idx2], 'b', label='D alpha')
        da_axis.set_ylabel('[a.u.]', fontsize=32)
        da_axis.tick_params(axis='y', labelsize=32)
        da_axis.legend(loc='best', fontsize=32)
        if threshold is not None:
            da_axis.axvline(threshold, color='r', linestyle='--')
    
    try:
        bottom_axis.set_xlim(time_lims)
    except:
        bottom_axis.set_xlim(timeslice)
    bottom_axis.set_xlabel('time [s]', fontsize=32)
    bottom_axis.tick_params(axis='x', labelsize=32)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('shot' + str(shot) + '_ch' + str(channel) + '_len' + 
                str(length) + '_' + str(direction) + '_velocity' + fname + 
                '.png', format='png', transparent=True)
    plt.close()

def plot_velocities_same_channel(
        BESdata, velocity_array_1, correlation_array_1, velocity_array_2, 
        correlation_array_2, time_points_1, num_slices_1, length_1, 
        time_points_2, num_slices_2, length_2, direction_1, direction_2, 
        label_1, label_2, row, column, h_lims, title, fname, dalpha=None, 
        bes_data=False, threshold=None, 
        n=7, end_lev=-1, v_lims_1=None, v_lims_2=None, freq_lims=[0.0, 200.0]):
    # plot comparison of two velocities of same channel 
    # this could be e.g. vertical and radial velocities 
    # can be with BES spectrogram (True/False) 
    # and/or Dalpha (set dalpha=[dalpha_time, dalpha_data])
    shot = BESdata.shot_number
    apdpos = BESdata.apdpos
    ncols = BESdata.ncols
    
    channel = ncols * row + column
    
    vels_ch_1, corrs_ch_1 = get_plausible_velocities(
        channel, row, column, velocity_array_1, correlation_array_1, apdpos, 
        num_slices_1, direction=direction_1)
    vels_ch_2, corrs_ch_2 = get_plausible_velocities(
        channel, row, column, velocity_array_2, correlation_array_2, apdpos, 
        num_slices_2, direction=direction_2)
    
    window_1 = np.hanning(length_1 + 1)
    velocity_line_1 = convolve(vels_ch_1, window_1)
    window_2 = np.hanning(length_2 + 1)
    velocity_line_2 = convolve(vels_ch_2, window_2)
    
    if bes_data is True:
        if dalpha is not None:
            figure, axes = plt.subplots(4, 1, sharex=True, figsize=(15, 15))
            vel1_axis = axes[0]
            vel2_axis = axes[1]
            bes_axis = axes[2]
            da_axis = axes[3]
            bottom_axis = axes[3]
            font_size = 24
        else:
            figure, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 15))
            vel1_axis = axes[0]
            vel2_axis = axes[1]
            bes_axis = axes[2]
            bottom_axis = axes[2]
            font_size = 32
    elif dalpha is not None:
        figure, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 15))
        vel1_axis = axes[0]
        vel2_axis = axes[1]
        da_axis = axes[2]
        bottom_axis = axes[2]
        font_size = 32
    else:
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 15))
        vel1_axis = axes[0]
        vel2_axis = axes[1]
        bottom_axis = axes[1]
        font_size = 32
    
    vel1_axis.set_title(str(shot) + ' ch ' + str(channel) + ', ' + title, 
                        fontsize=font_size)
    vel1_axis.axhline(0.0, linestyle='--', color='grey')
    vel1_axis.scatter(time_points_1, (1.0e-3 * np.asarray(vels_ch_1)), 
        c=(np.asarray(corrs_ch_1)), marker='.', cmap=plt.get_cmap('spring_r'))
    vel1_axis.plot(time_points_1, 1.0e-3 * velocity_line_1, 'k', label=label_1)
    vel1_axis.set_ylabel(direction_1 + ' velocity [km/s]', fontsize=font_size)
    vel1_axis.tick_params(axis='y', labelsize=font_size)
    vel1_axis.legend(loc='upper right', fontsize=font_size)
    try:
        vel1_axis.set_ylim(v_lims_1)
    except:
        pass
    vel2_axis.axhline(0.0, linestyle='--', color='grey')
    vel2_axis.scatter(time_points_2, (1.0e-3 * np.asarray(vels_ch_2)), 
        c=(np.asarray(corrs_ch_2)), marker='.', cmap=plt.get_cmap('spring_r'))
    vel2_axis.plot(time_points_2, 1.0e-3 * velocity_line_2, 'k', label=label_2)
    vel2_axis.set_ylabel(direction_2 + ' velocity [km/s]', fontsize=font_size)
    vel2_axis.tick_params(axis='y', labelsize=font_size)
    vel2_axis.legend(loc='upper right', fontsize=font_size)
    try:
        vel2_axis.set_ylim(v_lims_2)
    except:
        pass
    if threshold is not None:
        vel1_axis.axvline(threshold, color='r')
        vel2_axis.axvline(threshold, color='r')
    if bes_data is True:
        bes_time = BESdata.time.cut
        fluct_data = BESdata.data.fluct
        start_idx = np.abs(bes_time - h_lims[0]).argmin()
        end_idx = np.abs(bes_time - h_lims[1]).argmin()
        freq, times, Sxx = sig.spectrogram(fluct_data[:, start_idx:end_idx], 
                                           fs=f_samp, nperseg=2 ** n, 
                                           scaling='spectrum')
        end_lev = int(np.ceil(np.log10(np.max(Sxx[:][:int(
                np.shape(freq)[0]/5), :]))))
        #start_lev = int(np.ceil(np.log10(np.min(Sxx[:][:int(
        #        np.shape(freq)[0]/5), :]))))
        levs = np.logspace(end_lev - 7, end_lev, num=16)
        bes_axis.contourf((times + bes_time[start_idx]), (1.0e-3 * freq[:14]), 
            Sxx[channel][:14,:], levs, cmap=plt.get_cmap('gnuplot2'), 
            norm=(colors.LogNorm()))
        bes_axis.set_ylim(freq_lims)
        bes_axis.tick_params(axis='y', labelsize=font_size)
        bes_axis.set_ylabel('frequency [kHz]', fontsize=font_size)
    if dalpha is not None:
        dalpha_time = dalpha[0]
        dalpha_data = dalpha[1]
        da_idx1 = (np.abs(dalpha_time - h_lims[0])).argmin()
        da_idx2 = (np.abs(dalpha_time - h_lims[1])).argmin()
        da_axis.plot(dalpha_time[da_idx1:da_idx2], 
                     dalpha_data[da_idx1:da_idx2], 'b', label='$D_{\\alpha}$')
        if threshold is not None:
            da_axis.axvline(threshold, color='r')
        da_axis.legend(loc='upper right', fontsize=font_size)
        da_axis.tick_params(axis='y', labelsize=font_size)
        da_axis.set_ylabel('$D_{\\alpha}$ [a.u.]', fontsize=font_size)
    bottom_axis.set_xlim(h_lims)
    bottom_axis.tick_params(axis='x', labelsize=font_size)
    bottom_axis.set_xlabel('time [s]', fontsize=font_size)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('shot' + str(shot) + '_ch' + str(channel) + '_velocities_' + 
                fname + '.png', format='png', transparent=True)
    plt.close()

def plot_velocities_multiple_channels(
        BESdata, channels, num_slices, velocities, correlations, time_points, 
        length, timeslice, title, fname, v_lims=None, h_lims=None, 
        threshold=None, direction='poloidal'):
    # plot the velocity results of a list of channels
    num_channels = len(channels)
    
    shot = BESdata.shot_number
    apdpos = BESdata.apdpos
    
    figure, axes = plt.subplots(num_channels, 1, sharex=True, figsize=(15, 20))
    axes[0].set_title(str(shot) + ', ' + direction + ' velocity, len' + 
        str(length) + ', ' + title, fontsize=32)
    for i in range(num_channels):
        ch = channels[i]
        row, column = get_row_and_column_from_channel(ch)
        vels_ch, corrs_ch = get_plausible_velocities(
            ch, row, column, velocities, correlations, apdpos, num_slices, 
            direction=direction)

        window = np.hanning(length + 1)
        velocity_line = convolve(vels_ch, window)
        axes[i].scatter(time_points, 1.0e-3 * np.asarray(vels_ch), 
                        c=np.asarray(corrs_ch), marker='.', 
                        cmap=plt.get_cmap('spring_r'))
        axes[i].plot(time_points, 1.0e-3 * velocity_line, 'k', 
                     label='ch '+str(ch))
        axes[i].set_ylabel('[km/s]', fontsize=26)
        axes[i].tick_params(axis='y', labelsize=26)
        if threshold is not None:
            axes[i].axvline(threshold, color='r')
        axes[i].axhline(0.0, linestyle='--', linewidth=0.8, color='k')
        axes[i].legend(loc='upper right', fontsize=26)
        try:
            axes[i].set_ylim(v_lims)
        except:
            pass

    axes[num_channels-1].tick_params(axis='x', labelsize=26)
    try:
        axes[num_channels-1].set_xlim(h_lims)
    except:
        axes[num_channels-1].set_xlim(timeslice)
    axes[num_channels-1].set_xlabel('time [s]', fontsize=26)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('shot' + str(shot) + '_len' + str(length) + '_' + direction + 
                '_vels_' + fname + '.png', format='png', transparent=True)
    plt.close()

def plot_vel_profile_with_dalpha(
        BESdata, velocity_array, time_points, corr_array, row, num_slices, 
        length, dalpha_time, dalpha_data, timeslice, title, fname, 
        direction='poloidal'):
    
    shot = BESdata.shot_number
    apdpos = BESdata.apdpos
    ncols = BESdata.ncols
    f_samp = BESdata.f_samp
    
    velocity_smoothed = []
    for col in range(ncols):
        ch = row * ncols + col
        
        vels_ch, corrs_ch = get_plausible_velocities(
            ch, row, col, velocity_array, corr_array, apdpos, num_slices, 
            direction=direction)
        window = np.hanning(length + 1)
        velocity_line = convolve(vels_ch, window)
        velocity_smoothed.append(velocity_line)

    da_idx1 = np.abs(dalpha_time - timeslice[0]).argmin()
    da_idx2 = np.abs(dalpha_time - timeslice[1]).argmin()
    dalpha_max = np.max(dalpha_data[da_idx1:da_idx2])
    dalpha_min = np.min(dalpha_data[da_idx1:da_idx2])
    velocity_smoothed = 1.0e-3 * np.asarray(velocity_smoothed)
    vel_min = np.nanmin(velocity_smoothed)
    vel_max = np.nanmax(velocity_smoothed)
    s = 0
    for t in range(0, num_slices, 100):
        s += 1
        fs_pos = []
        for col in range(ncols):
            ch = row * ncols + col
            flux_surf_pos = map_channel_to_flux_surfaces(shot, time_points[t], 
                                                         apdpos, ch)
            fs_pos.append(flux_surf_pos)
        file_name = 'shot' + str(shot) + '_velocity_profile_with_dalpha_row' + str(
                row) + '_' + fname + '_' + str(s).zfill(6) + '.png'
        figure, axes = plt.subplots(2, 1, figsize=(8.5, 8.5))
        axes[0].plot(fs_pos, velocity_smoothed[:, t], 'ko--')
        axes[0].set_xlabel('flux surface position', fontsize=18)
        axes[0].set_ylabel('vertical velocity [km/s]', fontsize=18)
        axes[0].set_ylim([vel_min, vel_max])
        axes[0].axvline(1.0, linestyle='--', color='r')
        axes[0].axhline(0.0, linestyle='--', color='b')
        axes[0].tick_params(axis='x', labelsize=16)
        axes[0].tick_params(axis='y', labelsize=16)
        axes[0].set_title((title + str('{:<09}'.format(round(time_points[t], 
            7)))), fontsize=18)
        axes[1].plot(dalpha_time, dalpha_data, 'b')
        axes[1].set_xlim(timeslice)
        axes[1].set_ylim([dalpha_min, dalpha_max])
        axes[1].axvspan(time_points[t] - ((length / 2) / f_samp), 
            time_points[t] + ((length / 2) / f_samp), alpha=0.7, color='r')
        #axes[1].axvline((time_points[t]), color='r')
        axes[1].set_xlabel('time [s]', fontsize=18)
        axes[1].set_ylabel('D alpha', fontsize=18)
        axes[1].tick_params(axis='x', labelsize=16)
        axes[1].tick_params(axis='y', labelsize=16)
        plt.savefig(file_name, format='png', transparent=True)
        plt.close()

def plot_two_velocity_locations_with_dalpha(
        BESdata, dalpha_time, dalpha_data, velocity_array, correlation_array, 
        time_points, num_slices, length, direction, channel_1, channel_2, 
        label_1, label_2, 
        h_lims, threshold=None, v_lims_1=None, v_lims_2=None):
    # haven't generalised this yet, I used it to make a plot for a poster
    shot = BESdata.shot_number
    apdpos = BESdata.apdpos
    
    row1, col1 = get_row_and_column_from_channel(channel_1)
    row2, col2 = get_row_and_column_from_channel(channel_2)
    
    vels_ch_1, corrs_ch_1 = get_plausible_velocities(
        channel_1, row1, col1, velocity_array, correlation_array, apdpos, 
        num_slices, direction=direction)
    vels_ch_2, corrs_ch_2 = get_plausible_velocities(
        channel_2, row2, col2, velocity_array, correlation_array, apdpos, 
        num_slices, direction=direction)
    
    window = np.hanning(length + 1)
    velocity_line_1 = convolve(vels_ch_1, window)
    velocity_line_2 = convolve(vels_ch_2, window)

    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(18, 15))
    axes[0].axhline(0.0, linestyle='--', color='grey')
    axes[0].scatter(time_points, (1.0e-3 * np.asarray(vels_ch_1)), 
        c=(np.asarray(corrs_ch_1)), marker='.', cmap=plt.get_cmap('cool'))
    axes[0].plot(time_points, 1.0e-3 * velocity_line_1, 'k', label=label_1)#'edge, $\psi=0.99$')
    axes[0].set_ylabel('$v_{BES}$ [km/s]', fontsize=46)
    axes[0].tick_params(axis='y', labelsize=46)
    axes[0].legend(loc='upper right', fontsize=44)
    try:
        axes[0].set_ylim(v_lims_1)
    except:
        pass
    axes[1].axhline(0.0, linestyle='--', color='grey')
    axes[1].scatter(time_points, (1.0e-3 * np.asarray(vels_ch_2)), 
        c=(np.asarray(corrs_ch_2)), marker='.', cmap=plt.get_cmap('cool'))
    axes[1].plot(time_points, 1.0e-3 * velocity_line_2, 'k', label=label_2)#'SOL, $\psi=1.04$')
    axes[1].set_ylabel('$v_{BES}$ [km/s]', fontsize=46)
    axes[1].set_xlim(h_lims)
    axes[1].tick_params(axis='y', labelsize=46)
    axes[1].legend(loc='upper right', fontsize=44)
    try:
        axes[1].set_ylim(v_lims_2)
    except:
        pass
    axes[2].set_xlim(h_lims)
    dalpha_idx1 = (np.abs(dalpha_time - h_lims[0])).argmin()
    dalpha_idx2 = (np.abs(dalpha_time - h_lims[1])).argmin()
    axes[2].plot(dalpha_time[dalpha_idx1:dalpha_idx2], 
        dalpha_data[dalpha_idx1:dalpha_idx2], 'b', label='$D_{\\alpha}$')
    if threshold is not None:
        axes[0].axvline(threshold, linestyle='--', color='r')
        axes[1].axvline(threshold, linestyle='--', color='r')
        axes[2].axvline(threshold, linestyle='--', color='r')
    axes[2].legend(loc='upper right', fontsize=44)
    #axes[2].set_xticks([0.24, 0.245, 0.25, 0.255, 0.26])
    axes[2].tick_params(axis='y', labelsize=46)
    axes[2].tick_params(axis='x', labelsize=44)
    axes[2].set_ylabel('a.u.', fontsize=46)
    axes[2].set_xlabel('time [s]', fontsize=46)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('shot' + str(shot) + '_two_vels_with_dalpha.png', 
                format='png', transparent=True)
    plt.close()
