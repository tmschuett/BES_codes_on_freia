from imports import *
from BESClass import *
from functions_BES import *

# collection of plotting functions for BES velocimetry checks

def plot_cctde_analysis_poloidal(
        BESdata, mid_time_point, slices, time_base, xcorr_length, 
        timepoint_number, row, column, fname, chan_list, fit_range):
    # plot the CCTDE analysis for BES poloidal velocity
    shot = BESdata.shot_number
    apdpos = BESdata.apdpos
    bes_time = BESdata.time.cut
    fluct_data = BESdata.data.fluct
    nrows = BESdata.nrows
    ncols = BESdata.ncols
    
    # function to plot the analysis of the TDE poloidal velocimetry method for 
    # a reference channel:
    refch = column + ncols * row
    distances = []
    time_dels = []
    weights = []
    
    colours = ['k', 'r', 'b', 'g']
    figure, axes = plt.subplots(2, 2, figsize=(15.5, 15.5))
    axes[(1, 0)].axvline(0.0, color='grey', linestyle='--')
    axes[(0, 0)].set_title(('shot ' + str(shot)), fontsize=32)
    for n in range(len(chan_list)):
        # the current channel for cross-correlation with reference channel
        ch = chan_list[n]
        # the cross-correlation for this channel
        xcorr = cross_corr(refch, ch, fluct_data, slices[0], slices[1])
        xcorr = xcorr.astype(float)
        axes[(1, 0)].plot((1.0e6 * np.asarray(time_base)), xcorr, 
            color=(colours[n]), label=('ch' + str(ch) + ' shifted with ch' + 
                   str(refch)))
        
        # the distance between channels 
        if ch == refch:
            distances.append(0.0)
        else:
            distances.append(
                ((ch - refch) / np.abs(ch - refch)) * np.sqrt(
                    np.abs(apdpos[ch][1] - apdpos[refch][1]) ** 2 + 
                    np.abs(apdpos[ch][0] - apdpos[refch][0]) ** 2))
        # the time delay of the peak
        time_del_peak = time_base[np.argmax(xcorr)]
        # the value of the cross-correlation at peak
        maxcorr_peak = np.max(xcorr)
        
        peak_idx = np.argwhere(time_base == time_del_peak)[0][0]
        idx_stop = peak_idx - int((fit_range / 2) * xcorr_length)
        if idx_stop < 0:
            idx_stop = 0

        idx_start = peak_idx + int((fit_range / 2) * xcorr_length)
        if idx_start > len(time_base) - 1:
            idx_start = len(time_base) - 1

        rev_time = np.flip(time_base[idx_stop:idx_start])
        rev_corr = np.flip(xcorr[idx_stop:idx_start])

        try:
            popt, pcov = curve_fit(
                gaussian, rev_time, rev_corr, p0=[
                    maxcorr_peak, time_del_peak, 
                    (rev_time[-1] - rev_time[0]) / 2.0])
            time_val = rev_time[np.argmax(gaussian(rev_time, *popt))]
            corr_val = np.max(gaussian(rev_time, *popt))
            axes[(1, 0)].plot((1.0e6 * np.flip(time_base)), 
                gaussian(np.flip(time_base), *popt), color=(colours[n]), 
                linestyle='--')
        except Exception as e:
            print(e)
            time_val = time_del_peak
            corr_val = maxcorr_peak

        axes[(1, 0)].plot((1.0e6 * time_val), corr_val, marker='*', 
                color=(colours[n]))
        axes[(1, 1)].plot(distances[n], 
            (1.0e6 * time_val), marker='*', color=(colours[n]), markersize=14)
        time_dels.append(time_val)
        weights.append(corr_val)

    axes[(1, 0)].legend(loc='lower left', fontsize=20)
    axes[(1, 0)].axhline(0.0, color='tab:gray', linestyle='--')
    axes[(1, 0)].set_xlabel('time delay [$\\mu$s]', fontsize=26)
    axes[(1, 0)].set_ylabel('cross-correlation coefficient', fontsize=26)
    axes[(1, 0)].tick_params(axis='x', labelsize=26)
    axes[(1, 0)].tick_params(axis='y', labelsize=26)
    
    # fit a straight line to chosen distance-time delay values, 
    # 1/gradient is velocity
    try:
        fit_params = np.polyfit(distances, time_dels, 1, w=(np.asarray(weights)))
        axes[(1, 1)].plot(distances, (1.0e6 * (fit_params[0] * 
             np.asarray(distances) + fit_params[1])), 'm', 
            label=('velocity: ' + str(round(1.0 / fit_params[0], 2))))
    except Exception as e:
        print(e)
        try:
            fit_params = np.polyfit(distances, time_dels, 1)
            axes[(1, 1)].plot(distances, (1.0e6 * (fit_params[0] * 
                 np.asarray(distances) + fit_params[1])), 'm', 
                label=('velocity: ' + str(round(1.0 / fit_params[0], 2))))
        except:
            pass

    axes[(0, 1)].set_title((str(round(mid_time_point, 7)) + ' poloidal, refch ' + 
        str(refch)), fontsize=32)
    axes[(1, 1)].legend(loc='best', fontsize=26)
    axes[(1, 1)].set_xlabel('distance between channels [m]', fontsize=26)
    axes[(1, 1)].set_ylabel('time delay [$\\mu$s]', fontsize=26)
    axes[(1, 1)].tick_params(axis='x', labelsize=26)
    axes[(1, 1)].tick_params(axis='y', labelsize=26)
    axes[(1, 1)].yaxis.set_label_position('right')
    axes[(1, 1)].yaxis.tick_right()
    
    for n in range(len(chan_list)):
        ch = chan_list[n]
        xcorr = cross_corr(refch, ch, fluct_data, slices[0], slices[1])
        ps_xcorr = np.abs(np.fft.fft(xcorr)) ** 2
        time_step_xcorr = time_base[1] - time_base[0]
        freqs_xcorr = np.fft.fftfreq(xcorr.size, time_step_xcorr)
        idx_xcorr = np.argsort(freqs_xcorr)
        axes[(0, 1)].plot((1.0e-3 * freqs_xcorr[idx_xcorr]), 
            (ps_xcorr[idx_xcorr]), color=(colours[n]), 
            label=('ch' + str(ncols * n + column) + ' shifted with ch' + 
                   str(ncols * row + column)))

    axes[(0, 1)].legend(loc='upper right', fontsize=18)
    axes[(0, 1)].set_xlabel('frequency [kHz]', fontsize=26)
    axes[(0, 1)].tick_params(axis='x', labelsize=26)
    axes[(0, 1)].tick_params(axis='y', labelsize=26)
    axes[(0, 1)].set_xlim([0.0, 250.0])
    axes[(0, 1)].set_yscale('log')
    axes[(0, 1)].set_ylim([1.0e-3, 1.0e3])
    axes[(0, 1)].yaxis.set_label_position('right')
    axes[(0, 1)].yaxis.tick_right()
    for n in range(len(chan_list)):
        ch = chan_list[n]
        axes[(0, 0)].plot((bes_time[slices[0]:slices[1]]), 
            (fluct_data[ch][slices[0]:slices[1]]), color=(colours[n]), 
            label=('ch' + str(ch)))

    axes[(0, 0)].legend(loc='best', fontsize=18)
    axes[(0, 0)].set_xlabel('time [s]', fontsize=26)
    axes[(0, 0)].set_ylabel('fluctuation data', fontsize=26)
    axes[(0, 0)].tick_params(axis='x', labelsize=26)
    axes[(0, 0)].tick_params(axis='y', labelsize=26)
    plt.savefig('shot' + str(shot) + '_ch' + str(refch) + 'len' + 
                str(xcorr_length) + '_' + fname + '_poloidal_xcorr_timept' + 
                str(timepoint_number) + '.png', format='png', transparent=True)
    plt.close()

def plot_cctde_analysis_radial(
        BESdata, mid_time_point, slices, time_base, xcorr_length, 
        timepoint_number, row, column, fname, radial_chans_list, fit_range):
    # plot the CCTDE analysis for BES radial velocity
    shot = BESdata.shot_number
    apdpos = BESdata.apdpos
    bes_time = BESdata.time.cut
    fluct_data = BESdata.data.fluct
    nrows = BESdata.nrows
    ncols = BESdata.ncols
    
    # function to plot the analysis of the TDE velocimetry method for 
    # a particular channel (for radial velocity)
    # the reference channel:
    refch = column + ncols * row
    distances = []
    time_dels = []
    weights = []
    
    colours = ['r', 'b', 'limegreen', 'fuchsia', 'darkorange', 
               'mediumpurple', 'olive', 'gold']
    figure, axes = plt.subplots(2, 2, figsize=(15.5, 15.5))
    axes[(1, 0)].axvline(0.0, color='grey', linestyle='--')
    axes[(0, 0)].set_title(('shot ' + str(shot)), fontsize=32)
    for n in range(len(radial_chans_list)):
        # the current channel for cross-correlation with reference channel
        ch = radial_chans_list[n]
        # the cross-correlation for this channel
        xcorr = cross_corr(refch, ch, fluct_data, slices[0], slices[1])
        xcorr = xcorr.astype(float)
        axes[(1, 0)].plot((1.0e6 * np.asarray(time_base)), xcorr, 
            color=(colours[n]), label=('ch' + str(ch) + ' shifted with ch' + 
                   str(refch)))
        
        # the distance between channels (z direction only)
        if ch == refch:
            distances.append(0.0)
        else:
            distances.append(((ch - refch) / np.abs(ch - refch)) * 
                             np.sqrt(np.abs(apdpos[ch][1] - 
                                            apdpos[refch][1]) ** 2 + np.abs(
                                                    apdpos[ch][0] - 
                                                    apdpos[refch][0]) ** 2))
        # the time delay of the peak
        time_del_peak = time_base[np.argmax(xcorr)]
        # the value of the cross-correlation at peak
        maxcorr_peak = np.max(xcorr).astype(float)
        
        peak_idx = np.argwhere(time_base == time_del_peak)[0][0]
        idx_stop = peak_idx - int((fit_range / 2) * xcorr_length)
        if idx_stop < 0:
            idx_stop = 0

        idx_start = peak_idx + int((fit_range / 2) * xcorr_length)
        if idx_start > len(time_base) - 1:
            idx_start = len(time_base) - 1

        rev_time = np.flip(time_base[idx_stop:idx_start])
        rev_corr = np.flip(xcorr[idx_stop:idx_start])
        try:
            popt, pcov = curve_fit(gaussian, rev_time, 
                                   rev_corr, p0=[maxcorr_peak, 
                                                 time_del_peak, 
                                                 (rev_time[-1] 
                                                 - rev_time[0]) 
                                                 / 2.0])
            time_val = rev_time[np.argmax(gaussian(rev_time, 
                                                   *popt))]
            corr_val = np.max(gaussian(rev_time, *popt))
            axes[(1, 0)].plot((1.0e6 * np.flip(time_base)), 
                gaussian(np.flip(time_base), *popt), color=(colours[n]), 
                linestyle='--')
        except Exception as e:
            print(e)
            time_val = time_del_peak
            corr_val = maxcorr_peak
        axes[(1, 0)].plot((1.0e6 * time_val), corr_val, marker='*', 
                color=(colours[n]))
        axes[(1, 1)].plot(distances[n], 
            (1.0e6 * time_val), marker='*', color=(colours[n]), markersize=14)
        time_dels.append(time_val)
        weights.append(corr_val)

    axes[(1, 0)].legend(loc='lower left', fontsize=20)
    axes[(1, 0)].axhline(0.0, color='tab:gray', linestyle='--')
    axes[(1, 0)].set_xlabel('time delay [$\\mu$s]', fontsize=26)
    axes[(1, 0)].set_ylabel('cross-correlation coefficient', fontsize=26)
    axes[(1, 0)].tick_params(axis='x', labelsize=26)
    axes[(1, 0)].tick_params(axis='y', labelsize=26)
    

    # fit a straight line to chosen distance-time delay values, 
    # 1/gradient is velocity
    try:
        fit_params = np.polyfit(distances, time_dels, 1, w=(np.asarray(weights)))
        axes[(1, 1)].plot(distances, (1.0e6 * (fit_params[0] * 
             np.asarray(distances) + fit_params[1])), 'grey', 
            label=('velocity: ' + str(round(1.0 / fit_params[0], 2))))
    except Exception as e:
        print(e)
        try:
            fit_params = np.polyfit(distances, time_dels, 1)
            axes[(1, 1)].plot(distances, (1.0e6 * (fit_params[0] * 
                 np.asarray(distances) + fit_params[1])), 'grey', 
                label=('velocity: ' + str(round(1.0 / fit_params[0], 2))))
        except:
            pass

    axes[(0, 1)].set_title((str(round(mid_time_point, 7)) + ' radial, refch ' + 
        str(refch)), fontsize=32)
    axes[(1, 1)].legend(loc='best', fontsize=26)
    axes[(1, 1)].set_xlabel('distance between channels [m]', fontsize=26)
    axes[(1, 1)].set_ylabel('time delay [$\\mu$s]', fontsize=26)
    axes[(1, 1)].tick_params(axis='x', labelsize=26)
    axes[(1, 1)].tick_params(axis='y', labelsize=26)
    axes[(1, 1)].yaxis.set_label_position('right')
    axes[(1, 1)].yaxis.tick_right()
    
    for n in range(len(radial_chans_list)):
        ch = radial_chans_list[n]
        xcorr = cross_corr(refch, ch, fluct_data, slices[0], slices[1])
        ps_xcorr = np.abs(np.fft.fft(xcorr)) ** 2
        time_step_xcorr = time_base[1] - time_base[0]
        freqs_xcorr = np.fft.fftfreq(xcorr.size, time_step_xcorr)
        idx_xcorr = np.argsort(freqs_xcorr)
        axes[(0, 1)].plot((1.0e-3 * freqs_xcorr[idx_xcorr]), 
            (ps_xcorr[idx_xcorr]), color=(colours[n]), 
            label=('ch' + str(ch) + ' shifted with ch' + 
                   str(refch)))

    axes[(0, 1)].legend(loc='upper right', fontsize=18)
    axes[(0, 1)].set_xlabel('frequency [kHz]', fontsize=26)
    axes[(0, 1)].tick_params(axis='x', labelsize=26)
    axes[(0, 1)].tick_params(axis='y', labelsize=26)
    axes[(0, 1)].set_xlim([0.0, 250.0])
    axes[(0, 1)].set_yscale('log')
    axes[(0, 1)].set_ylim([1.0e-3, 1.0e3])
    axes[(0, 1)].yaxis.set_label_position('right')
    axes[(0, 1)].yaxis.tick_right()
    for n in range(len(radial_chans_list)):
        ch = radial_chans_list[n]
        axes[(0, 0)].plot((bes_time[slices[0]:slices[1]]), 
            (fluct_data[ch][slices[0]:slices[1]]), color=(colours[n]), 
            label=('ch' + str(ch)))

    axes[(0, 0)].legend(loc='best', fontsize=18)
    axes[(0, 0)].set_xlabel('time [s]', fontsize=26)
    axes[(0, 0)].set_ylabel('fluctuation data', fontsize=26)
    axes[(0, 0)].tick_params(axis='x', labelsize=26)
    axes[(0, 0)].tick_params(axis='y', labelsize=26)
    plt.savefig('shot' + str(shot) + '_ch' + str(refch) + 'len' + 
                str(xcorr_length) + '_' + fname + '_radial_xcorr_timept' + 
                str(timepoint_number) + '.png', format='png', transparent=True)
    plt.close()

def plot_velocity_distribution(
        BESdata, velocity_array, row, column, num_slices, length, fname, 
        title, bin_size=5.0e3, direction='poloidal'):
    # plots a histogram of the velocities for this channel
    shot = BESdata.shot_number
    apdpos = BESdata.apdpos
    ncols = BESdata.ncols
    nrows = BESdata.nrows
    f_samp = BESdata.f_samp
    
    channel = ncols * row + column
    
    if direction == 'vertical' or direction == 'poloidal':
        if row == 0:
            vels_lim = np.abs((apdpos[channel+ncols][1] - 
                               apdpos[channel][1]) * f_samp)
        elif row == nrows-1:
            vels_lim = np.abs((apdpos[channel-ncols][1] - 
                               apdpos[channel][1]) * f_samp)
        else:
            vels_lim_1 = np.abs((apdpos[(channel+ncols)][1] - 
                                        apdpos[channel][1]) * f_samp)
            vels_lim_2 = np.abs((apdpos[(channel-ncols)][1] - 
                                        apdpos[channel][1]) * f_samp)
            vels_lim = min(vels_lim_1, vels_lim_2)
    elif direction == 'radial':
        if column == 0:
            vels_lim = np.abs((apdpos[channel+1][0] - 
                               apdpos[channel][0]) * f_samp)
        elif column == ncols-1:
            vels_lim = np.abs((apdpos[channel-1][0] - 
                               apdpos[channel][0]) * f_samp)
        else:
            vels_lim_1 = np.abs((apdpos[(channel+1)][0] - 
                                        apdpos[channel][0]) * f_samp)
            vels_lim_2 = np.abs((apdpos[(channel-1)][0] - 
                                        apdpos[channel][0]) * f_samp)
            vels_lim = min(vels_lim_1, vels_lim_2)
            
    num_edges = int(25.0e3/bin_size)
    bin_edges = [-(vels_lim + 10.0e3), -vels_lim]
    new_edges = range(-num_edges, num_edges + 1)
    for edge in new_edges:
        bin_edges.append(edge * bin_size)
    bin_edges.extend([vels_lim, (vels_lim + 10.0e3), (vels_lim + 20.0e3)])
    bin_edges = np.asarray(bin_edges)
        
    vels_ch = []
    num_minlim = 0
    num_inf = 0
    num_poslim = 0
    for i in range(num_slices):
        vel_ch = velocity_array[i][row][column]
        if vel_ch < -vels_lim:
            vels_ch.append(-(vels_lim + 5.0e3))
            num_minlim += 1
        elif math.isinf(vel_ch):
            vels_ch.append(vels_lim + 15.0e3)
            num_inf += 1
        elif vel_ch > vels_lim:
            vels_ch.append(vels_lim + 5.0e3)
            num_poslim += 1
        else:
            vels_ch.append(vel_ch)
     
    vels = np.asarray(vels_ch)
    using_search_sorted = np.bincount(np.searchsorted(bin_edges, vels))
    
    bin_count = np.zeros(np.shape(bin_edges)[0] - 1)
    for j in range(np.amax(np.searchsorted(bin_edges, vels))):
        bin_count[j] = using_search_sorted[j+1]    
        
    rounded_bins = np.round(1.0e-3*bin_edges, decimals=1)
    bin_labels = []
    for i in range(len(rounded_bins) - 1):
        bin_labels.append(str(int(rounded_bins[i])) + ' to ' + 
                        str(int(rounded_bins[i + 1])))
    
    bin_labels[0] = 'O.o.B.'
    bin_labels[-2] = 'O.o.B.'
    bin_labels[-1] = r'$\infty$'
    bin_count = bin_count.tolist()
        
    do_not_plot_excluded = False
    num_bins = np.shape(bin_count)[0]
    excluded_bins = [bin_count[0], bin_count[-2], bin_count[-1]]
    excluded_pos = [0, num_bins - 2, num_bins - 1]
    if num_minlim == 0:
        bin_count = bin_count[1:]
        bin_labels = bin_labels[1:]
        excluded_pos[-1] = excluded_pos[-1] - 1
        excluded_pos[-2] = excluded_pos[-2] - 1
        excluded_bins = excluded_bins[1:]
        excluded_pos = excluded_pos[1:]
    if num_inf == 0:
        bin_count = bin_count[:-1]
        bin_labels = bin_labels[:-1]
        excluded_bins = excluded_bins[:-1]
        excluded_pos = excluded_pos[:-1]
        if num_poslim == 0:
            bin_count = bin_count[:-1]
            bin_labels = bin_labels[:-1]
            do_not_plot_excluded = True
            excluded_bins = excluded_bins[:-1]
            excluded_pos = excluded_pos[:-1]
    else:
        if num_poslim == 0:
            bin_count.pop(-2)
            bin_labels.pop(-2)
            excluded_bins.pop(-2)
            excluded_pos = excluded_pos[:-1]
    bin_pos = np.arange(np.shape(bin_count)[0])
    plt.figure(figsize=(10, 10))
    plt.bar(bin_pos, height=bin_count, color='orangered',
            label='velocities within bounds')
    if do_not_plot_excluded is False:
        plt.bar(excluded_pos, height=excluded_bins, color='darkgrey',
                label='velocities out of bounds')
    plt.ylabel('count', fontsize=20)
    plt.legend(loc='best', fontsize=20)
    plt.xticks(bin_pos, bin_labels, fontsize=20, rotation=45)
    plt.yticks(fontsize=20)
    plt.title(str(shot) + ' ch ' + str(channel) + ' ' + direction + 
              ' velocity distr. [km/s], ' + title, fontsize=24)
    plt.savefig('shot' + str(shot) + '_ch' + str(channel) + '_' + direction + 
                '_velocity_distribution_' + fname + '.png', format='png', 
                transparent=True)
    plt.close()

def plot_velocity_comparison_same_axis(
        BESdata, velocity_array_1, velocity_array_2, corr_vals_1, corr_vals_2, 
        num_slices_1, num_slices_2, time_points, row, column, length_1, 
        length_2, label_1, label_2, fname_part, threshold=None, v_lims=None, 
        h_lims=None, direction='poloidal', line_only=True):
    # plot the results of two velocity methods for the same channel on top of
    # each other
    shot = BESdata.shot_number
    apdpos = BESdata.apdpos
    ncols = BESdata.ncols
    
    channel = ncols * row + column
    vels_1, corrs_1 = get_plausible_velocities(
        channel, row, column, velocity_array_1, corr_vals_1, apdpos, 
        num_slices_1, direction=direction)
    vels_2, corrs_2 = get_plausible_velocities(
        channel, row, column, velocity_array_2, corr_vals_2, apdpos, 
        num_slices_2, direction=direction)

    window_1 = np.hanning(length_1 + 1)
    window_2 = np.hanning(length_2 + 1)
    velocity_line_1 = convolve(vels_1, window_1)
    velocity_line_2 = convolve(vels_2, window_2)
    
    font_size = 32
    
    plt.figure(figsize=(15, 8))
    plt.title(str(shot) + ' channel ' + str(channel) + ', ' + direction + 
               ' velocity comparison', fontsize=font_size)
    if line_only is False:
        plt.scatter(time_points, 1.0e-3 * np.asarray(vels_1), c='salmon', 
                    marker='.')
        plt.scatter(time_points, 1.0e-3 * np.asarray(vels_2), c='royalblue', 
                    marker='.')
    plt.plot(time_points, 1.0e-3 * velocity_line_1, color='r', linewidth=2, 
             label=label_one)
    plt.plot(time_points, 1.0e-3 * velocity_line_2, color='b', linewidth=2, 
             label=label_two)
    plt.axhline(0.0, color='grey', linestyle='--', linewidth=1.5)
    plt.ylabel(direction + ' velocity [km/s]', fontsize=font_size)
    try:
        plt.xlim(h_lims)
    except:
        pass
    try:
        plt.ylim(v_lims)
    except:
        pass
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    if threshold is not None:
        plt.axvline(threshold, color='grey', linewidth=1.5)
    plt.xlabel('time [s]', fontsize=font_size)
    plt.legend(loc='best', fontsize=font_size)
    plt.tight_layout()
    fname = 'shot' + str(shot) + '_ch' + str(channel) + fname_part + '.png'
    plt.savefig(fname, format='png', transparent=True)
    plt.close()

def plot_fit_type_velocities(
        BESdata, dalpha_time, dalpha_data, list_of_velocities, list_of_times, 
        list_of_labels, list_of_correlations, length, num_slices, direction, 
        row, column, h_lims, ylabel, threshold=None):
    shot = BESdata.shot_number
    apdpos = BESdata.apdpos
    ncols = BESdata.ncols
    channel = ncols * row + column
    cmaps = ['Purples', 'Greens', 'Reds', 'Blues', 'Oranges']
    colours = ['tab:purple', 'tab:green', 'tab:red', 'tab:blue', 'tab:orange']
    
    window = np.hanning(length + 1)
    
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 15))
    axes[0].set_title((str(shot) + ' ch ' + str(channel) + ', comparing velocimetry fits'), 
        fontsize=32)
    axes[0].axhline(0.0, linestyle='--', color='grey')
    axes[1].axhline(0.0, linestyle='--', color='grey')
    for fit_type in range(len(list_of_velocities)):
        vels_array = list_of_velocities[fit_type]
        corrs_array = list_of_correlations[fit_type]
        times_array = list_of_times[fit_type]
        vels_ch, corrs_ch = get_plausible_velocities(
            channel, row, column, vels_array, corrs_array, apdpos, num_slices, 
            direction=direction)
        velocity_line = convolve(vels_ch, window)
        axes[0].scatter(times_array, (1.0e-3 * np.asarray(vels_ch)), 
            c=(np.asarray(corrs_ch)), marker='.', 
            cmap=plt.get_cmap(cmaps[fit_type]))
        axes[0].plot(times_array, 1.0e-3 * velocity_line, 
            color=colours[fit_type], label=list_of_labels[fit_type])
        axes[1].plot(times_array, 1.0e-3 * velocity_line, 
            color=colours[fit_type], label=list_of_labels[fit_type])
    axes[0].set_ylabel(ylabel + ' [km/s]', fontsize=32)
    axes[0].tick_params(axis='y', labelsize=32)
    axes[1].set_ylabel(ylabel + ' [km/s]', fontsize=32)
    axes[1].tick_params(axis='y', labelsize=32)
    axes[1].legend(loc='upper right', fontsize=26)
    axes[0].set_ylim([-30.0, 30.0])
    axes[1].set_ylim([-30.0, 30.0])
    dalpha_idx1 = (np.abs(dalpha_time - h_lims[0])).argmin()
    dalpha_idx2 = (np.abs(dalpha_time - h_lims[1])).argmin()
    axes[2].plot(dalpha_time[dalpha_idx1:dalpha_idx2], 
        dalpha_data[dalpha_idx1:dalpha_idx2], 'b', 
        label='tangential midplane $D_{\\alpha}$')
    if threshold is not None:
        axes[0].axvline(threshold, linestyle='--', color='k')
        axes[1].axvline(threshold, linestyle='--', color='k')
        axes[2].axvline(threshold, linestyle='--', color='k')
    axes[2].legend(loc='upper right', fontsize=32)
    axes[2].tick_params(axis='y', labelsize=32)
    axes[2].tick_params(axis='x', labelsize=32)
    axes[2].set_ylabel('$D_{\\alpha}$ [a.u.]', fontsize=32)
    axes[2].set_xlabel('time [s]', fontsize=32)
    axes[2].set_xlim(h_lims)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('shot' + str(shot) + '_ch' + str(channel) + 
                '_comparing_velocimetry_fit_types.png', format='png', 
                transparent=True)
    plt.close()

def plot_poloidal_radial_velocity_correlation(
        BESdata, row, column, mid_time_points, pol_vels, pol_corr, rad_vels, 
        rad_corr, xcorr_length_orig, timeslice, num_slices, 
        xcorr_length_new=5000, threshold=None):
    shot = BESdata.shot_number
    apdpos = BESdata.apdpos
    ncols = BESdata.ncols
    channel = column + ncols * row
    pol_vels_ch, pol_corrs_ch = get_plausible_velocities(
        channel, row, column, pol_vels, pol_corr, apdpos, num_slices, 
        direction='poloidal')
    rad_vels_ch, rad_corrs_ch = get_plausible_velocities(
        channel, row, column, rad_vels, rad_corr, apdpos, num_slices, 
        direction='radial')
    window = np.hanning(xcorr_length_orig + 1)
    pol_velocity_line = convolve(pol_vels_ch, window)
    rad_velocity_line = convolve(rad_vels_ch, window)
    
    pol_vel_line_interp = interp1d(
        mid_time_points[~np.isnan(pol_velocity_line)], 
        pol_velocity_line[~np.isnan(pol_velocity_line)], 
        bounds_error=False, fill_value=0.0)
    pol_vel_line = pol_vel_line_interp(mid_time_points)
    
    rad_vel_line_interp = interp1d(
        mid_time_points[~np.isnan(rad_velocity_line)], 
        rad_velocity_line[~np.isnan(rad_velocity_line)], 
        bounds_error=False, fill_value=0.0)
    rad_vel_line = rad_vel_line_interp(mid_time_points)
    
    idx1_vels = (np.abs(mid_time_points - timeslice[0])).argmin()
    idx2_vels = (np.abs(mid_time_points - timeslice[1])).argmin()
    num_slices_corrs = int((idx2_vels - idx1_vels) - xcorr_length_new) + 1
    
    rad_pol_corrs_list = []
    corrs_times_list = []
    for j in range(num_slices_corrs):
        try:
            index1 = idx1_vels + j
            index2 = index1 + xcorr_length_new
            pol_rad_corr = np.fft.ifft(np.fft.fft(
                    pol_vel_line[index1:index2]) * np.conj(
            np.fft.fft(rad_vel_line[index1:index2]))) / (len(
                    rad_vel_line[index1:index2]) * np.sqrt(
                    np.mean(pol_vel_line[index1:index2] ** 2) * np.mean(
                            rad_vel_line[index1:index2] ** 2)))
            rad_pol_corrs_list.append(pol_rad_corr[0])
            corrs_times_list.append(0.5 * (mid_time_points[index1] + 
                                           mid_time_points[index2]))
            #vert_rad_corr_mid = correlation_between_signals(
            #        pol_velocity_line[index1:index2], 
            #        rad_velocity_line[index1:index2])
        except Exception as e:
            print(e)
    figure, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    axes[0].set_title(str(shot) + ' , ch ' + str(channel) + 
                      ' poloidal-radial velocity correlation (over ' + 
                      str(round(1000*2e6*xcorr_length_new), 2) + 'ms)', 
                      fontsize=32)
    axes[0].plot(mid_time_points, 1.0e-3 * pol_velocity_line, 'b', 
                 linewidth=2, label='poloidal')
    axes[0].plot(mid_time_points, 1.0e-3 * rad_velocity_line, 'r', 
                 linewidth=2, label='radial')
    axes[0].axhline(0.0, color='grey', linestyle='--', linewidth=1.5)
    axes[0].legend(loc='best', fontsize=32)
    axes[0].tick_params(axis='y', labelsize=32)
    axes[0].set_ylabel('velocity [km/s]', fontsize=32)
    axes[1].plot(corrs_times_list, rad_pol_corrs_list, 'k', linewidth=2.5, 
                 label='$\\theta - r$ correlation')
    axes[1].legend(loc='best', fontsize=32)
    axes[1].set_ylim([-1.0, 1.0])
    axes[1].axhline(0.0, color='grey', linestyle='--', linewidth=1.5)
    axes[1].tick_params(axis='x', labelsize=32)
    axes[1].tick_params(axis='y', labelsize=32)
    axes[1].set_xlabel('time [s]', fontsize=32)
    axes[1].set_ylabel('correlation', fontsize=32)
    if threshold is not None:
        axes[0].axvline(threshold, linestyle=':', color='grey', linewidth=1.5)
        axes[1].axvline(threshold, linestyle=':', color='grey', linewidth=1.5)
    axes[1].set_xlim(timeslice)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('shot' + str(shot) + '_ch' + str(channel) + 
                '_poloidal_radial_velocity_correlation_interpolated.png', 
                format='png', transparent=True)
    plt.close()
