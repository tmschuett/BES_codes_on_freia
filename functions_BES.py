# collect functions for BES calculations
from imports import *
from BESClass import *

def cross_corr(refch, ch, fluct_data, slice_idx1, slice_idx2):
    # calculate the cross-correlation between two BES channels
    # used in velocimetry
    ref_data = fluct_data[refch, slice_idx1:slice_idx2]
    ch_data = fluct_data[ch, slice_idx1:slice_idx2]
    cross_corr_coeff = np.fft.ifft(np.fft.fft(ref_data) * np.conj(
            np.fft.fft(ch_data))) / (len(ch_data) * np.sqrt(
                    np.mean(ref_data ** 2) * np.mean(ch_data ** 2)))
    return np.roll(cross_corr_coeff, int(0.5 * len(cross_corr_coeff)))

def calc_zero_lag_cross_corr(refch, ch, fluct_data, slice_idx1, slice_idx2):
    # calculate the cross-correlation at zero time lag
    ref_data = fluct_data[refch, slice_idx1:slice_idx2]
    ch_data = fluct_data[ch, slice_idx1:slice_idx2]
    cross_corr_coeff = np.fft.ifft(np.fft.fft(ref_data)*np.conj(
            np.fft.fft(ch_data)))/(len(ch_data)*np.sqrt(
                    np.mean(ref_data**2)*np.mean(ch_data**2)))
    return cross_corr_coeff[0]

def calc_velocity_trace_for_ch(apdpos, velocity_array, time_points, 
                               row, column, num_slices, xcorr_length, 
                               direction='poloidal', ncols=8, nrows=4, 
                               f_samp=2.0e6):
    # calculate the convolved trace for the velocity of one channel
    channel = ncols*row+column
    
    if direction == 'vertical' or direction == 'poloidal':
        if row == 0:
            vels_lim = np.abs((apdpos[channel+ncols][1] - 
                               apdpos[channel][1]) * f_samp)
        elif row == (nrows-1):
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
        elif column == (ncols-1):
            vels_lim = np.abs((apdpos[channel-1][0] - 
                               apdpos[channel][0]) * f_samp)
        else:
            vels_lim_1 = np.abs((apdpos[(channel+1)][0] - 
                                        apdpos[channel][0]) * f_samp)
            vels_lim_2 = np.abs((apdpos[(channel-1)][0] - 
                                        apdpos[channel][0]) * f_samp)
            vels_lim = min(vels_lim_1, vels_lim_2)
       
    vels_ch = []
    for i in range(num_slices):
        vel_ch = velocity_array[i][row][column]
        if math.isinf(vel_ch) is False and np.abs(
                vel_ch) < vels_lim:
            vels_ch.append(vel_ch)
        else:
            vels_ch.append(np.NaN)
    window = np.hanning(xcorr_length + 1)
    velocity_line = 1.0e-3*convolve(vels_ch, window)
    
    return velocity_line

def get_plausible_velocities(row, column, velocities, correlations, 
                             apdpos, num_slices, direction='poloidal', 
                             ncols=8, nrows=4, f_samp=2.0e6):
    # remove all velocities above limit determined by sampling frequency and 
    # channel separation
    channel = ncols*row + column
    
    if direction == 'vertical' or direction == 'poloidal':
        if row == 0:
            vels_lim = np.abs((apdpos[channel+ncols][1] - 
                               apdpos[channel][1]) * f_samp)
        elif row == (nrows-1):
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
        elif column == (ncols-1):
            vels_lim = np.abs((apdpos[channel-1][0] - 
                               apdpos[channel][0]) * f_samp)
        else:
            vels_lim_1 = np.abs((apdpos[(channel+1)][0] - 
                                        apdpos[channel][0]) * f_samp)
            vels_lim_2 = np.abs((apdpos[(channel-1)][0] - 
                                        apdpos[channel][0]) * f_samp)
            vels_lim = min(vels_lim_1, vels_lim_2)
            
    vels_ch = []
    corrs_ch = []
    for j in range(num_slices):
        vel_ch = velocities[j][row][column]
        corr_ch = correlations[j][row][column]
        if math.isinf(vel_ch) is False and np.abs(vel_ch) < vels_lim:
            vels_ch.append(vel_ch)
            corrs_ch.append(corr_ch)
        else:
            vels_ch.append(np.NaN)
            corrs_ch.append(np.NaN)
            
    return vels_ch, corrs_ch

def map_channel_to_flux_surfaces(shot, timepoint, apdpos, channel):
    equilib = eq.equilibrium(device='MAST', shot=shot, time=timepoint)
    
    equilib_interp = interp2d(equilib.R, equilib.Z, np.sqrt(equilib.psiN), 
                              bounds_error=False, fill_value=0.0)
    equilib_val = equilib_interp(apdpos[channel, 0], apdpos[channel, 1])
    
    return equilib_val[0]

def correlation_between_signals(signal_1, signal_2):
    cross_corr_coeff = np.fft.ifft(np.fft.fft(signal_1) * np.conj(
            np.fft.fft(signal_2))) / (len(signal_2) * np.sqrt(
                    np.mean(signal_1 ** 2) * np.mean(signal_2 ** 2)))
    return np.roll(cross_corr_coeff, int(0.5 * len(cross_corr_coeff)))

def calc_radial_chans_from_equilibrium(equilib_R, equilib_Z, equilib_psi_t, 
                                       apdpos, BESdata):
    nrows = BESdata.nrows
    ncols = BESdata.ncols
    nchan = BESdata.nchan
    columns = BESdata.column_list
    # initialise array of chosen channels (out of row) for each reference 
    # channel
    # set channel numbers to -2 and only fill if a channel is chosen
    eq_radial_chans = np.full((nchan, ncols), -2)
    
    # finding the best flux surface
    # Point object of channel 0 (R,z)
    point0 = geom.Point(apdpos[0][0], apdpos[0][1])
    # try first with psi=1 (should be separatrix, but isn't always..)
    sep_pos = 1.0
    CS = plt.contour(equilib_R, equilib_Z, equilib_psi_t, np.array([sep_pos]))
    dat = CS.allsegs[0][0]
    line = geom.LineString(dat)
    # project location of channel 0 onto this flux surface
    point_on_line = line.interpolate(line.project(point0))
    while point_on_line.x < 1.0 or np.abs(point_on_line.y) > 0.5:
        # if this point on the flux surface has R<1m or |z|>0.5m, add small 
        # increment and try again
        sep_pos = sep_pos + 0.0001
        CS = plt.contour(equilib_R, equilib_Z, equilib_psi_t, 
                         np.array([sep_pos]))
        dat = CS.allsegs[0][0]
        line = geom.LineString(dat)
        point_on_line = line.interpolate(line.project(point0))
    # this is probably not very efficient, but it has worked for all my 
    # problem cases so far
    
    # once the best flux surface has been found, cycle through all channels 
    # as reference channel
    for refch in range(nchan):
        # Point object of reference channel
        point = geom.Point(apdpos[refch][0], apdpos[refch][1])
        # project onto flux surface
        point_on_line = line.interpolate(line.project(point))
        # calculate the slope and intercept of the line from reference 
        # channel to FS
        slope = (point_on_line.y - apdpos[refch][1])/(point_on_line.x - 
                                                      apdpos[refch][0])
        intercept = point_on_line.y - (slope*point_on_line.x)
        # for each column location
        for col_no in range(ncols):
            col = columns[col_no]
            rem_vals = []
            for ch in col:
                # cycle through the channels in this column to find the 
                # vertical distance between the channel location and the line
                rem_val = apdpos[ch][1] - ((slope*apdpos[ch][0]) + intercept)
                rem_vals.append(rem_val)
            # find the channel of that column which has the smallest distance 
            # to the line
            abs_rem_vals = np.abs(np.asarray(rem_vals))
            min_rem_val = np.min(abs_rem_vals)
            min_rem_val_idx = np.argmin(abs_rem_vals)
            if min_rem_val < 0.012:
                # choose that channel only if the distance is below 1.2cm
                # otherwise leave at -2
                eq_radial_chans[refch, col_no] = col[min_rem_val_idx]
    plt.close()
    return eq_radial_chans

def calc_kspec(spec, xvec, odd=0, decimals=1, antialias=1., direction=1.):
    # from I. Cziegler
    xsort = np.sort(xvec)
    dxsort = np.sort(np.diff(xsort))
    dx = dxsort[0]
    n = 0
    while dx == 0:
        n = n + 1
        dx = dxsort[n]
    kmax = np.pi / dx * antialias
    kmin = np.floor((10 ** decimals) * np.pi / 
                    (xvec.max() - xvec.min()) / 2) / (10 ** decimals)
    while kmin == 0:
        decimals=decimals + 1
        kmin = np.floor((10 ** decimals) * np.pi / 
                        (xvec.max() - xvec.min()) / 2) / (10 ** decimals)
    klen = int(2 * np.around(kmax / kmin) + odd)
    k_arr = np.linspace(-kmax, kmax, klen)
    fcomps = np.exp(1j * direction * np.outer(xvec, k_arr))
    kfspec = np.dot(spec, fcomps)
    return kfspec, k_arr

# function to calculate all kinds of coherence-related parameters
def calc_phases(BESdata, timeslice, freqrange, pickfreq, n=7, refch=4, 
                coherence_limit=0.65):
    bes_time = BESdata.time.cut
    nchan = BESdata.nchan
    f_samp = BESdata.f_samp
    
    start_index = (np.abs(bes_time - timeslice[0])).argmin()
    stop_index = (np.abs(bes_time - timeslice[1])).argmin()
    bes_timeslice = bes_time[start_index:stop_index]
    bes_dataslice = BESdata.data.fluct[:,start_index:stop_index]
    # the number of data points per segment (set with parameter n)
    numperseg = 2**n
    # the number of data points in the time slice
    numperslice = np.shape(bes_timeslice)[0]
    # the number of segments in this time slice (used for averaging - 
    # higher number gives lower uncertainties)
    numsegments = numperslice/numperseg
    # the coherence is calculated with respect to a reference channel, 
    # so calculate the auto-spectral density for this
    bes_ref_fft = sig.csd(bes_dataslice[refch], bes_dataslice[refch], 
                        fs=f_samp, nperseg=numperseg, window='hann')
    # the parameters calculated and returned in this function:
    # the average phase (averaged over a region determined by the frequency
    # range and coherence limit)
    average_phase = np.zeros(nchan) 
    phase_errors = np.zeros(nchan)
    # the gradient (of phase in frequency space)
    phase_gradients = np.zeros(nchan)
    gradient_errors = np.zeros(nchan)
    # the maximum level of coherence between the two channels
    max_coherence = np.zeros(nchan)
    # the average frequency where coherent (weighted by coherence)
    frequencies = np.zeros(nchan)
    # for most calculations the values at a particular frequency are 
    # important (pickfreq here)
    phase_for_specific_freq = np.zeros(nchan)
    phase_error_for_specific_freq = np.zeros(nchan)
    for i in range(nchan):
        bes_auto_fft = sig.csd(bes_dataslice[i], bes_dataslice[i], 
                               fs=f_samp, nperseg=numperseg, 
                               window='hann')
        bes_x_fft = sig.csd(bes_dataslice[i], bes_dataslice[refch], 
                            fs=f_samp, nperseg=numperseg, 
                            window='hann')
        startidx = np.argwhere(bes_x_fft[0] > freqrange[0])[0][0]
        stopidx = np.argwhere(bes_x_fft[0] > freqrange[1])[0][0]
        # cut to the given frequency range (and express frequency in kHz)
        frequency = 1.0e-3*bes_x_fft[0][startidx : stopidx]
        # formulae for finding the coherence and phase between two channels
        coherence = ((np.abs(bes_x_fft[1])/np.sqrt(np.abs(bes_ref_fft[1])*
                             np.abs(bes_auto_fft[1])))[startidx : stopidx])
        phase = np.angle(bes_x_fft[1], deg=True)[startidx : stopidx]
        # determine where the coherence fulfils the requirements 
        # (with cohlimit)
        # different cases to account for identical channels returning 
        # coherence = 1 and cases where coherence not above limit
        try:
            coherence_start = np.argwhere(coherence 
                                          > coherence_limit)[0][0]
            coherence_stop = np.argwhere(coherence[coherence_start:] 
                                         < coherence_limit)[0][0]+coherence_start
            if coherence_start == coherence_stop:
                coherence_stop = coherence_stop + 1
        except:
            try:
                coherence_start = np.argwhere(coherence 
                                              > coherence_limit)[0][0]
                coherence_stop = np.argwhere(coherence 
                                             > coherence_limit)[-1][0]
                coherence_stop = coherence_stop + 1
            except:
                coherence_start = 0
                coherence_stop = 0
        # cut the frequencies, coherences and phases to only where the 
        # signals are above the coherence threshold
        new_frequency = frequency[coherence_start : coherence_stop]
        new_phase = phase[coherence_start : coherence_stop]
        new_coherence = coherence[coherence_start : coherence_stop]
        max_coherence[i] = np.max(np.abs(coherence))
        phase_for_specific_freq[i] = np.interp(pickfreq, frequency, phase)
        coherence_for_freq = np.interp(pickfreq, frequency, coherence)
        # using Eq 3 from S. Freethy, Phys. Plasmas 25, 055903 (2018) 
        # for error on phases 
        phase_err = np.sqrt(1.0/(2.0*numsegments))*(
                (1.0/coherence_for_freq**2) - 1.0)*(180.0/np.pi)
        phase_error_for_specific_freq[i] = phase_err
        # this 'unwraps' the phases, getting rid of the phase jumps 
        # (greater than pi)
        try:
            shifted_phase = [new_phase[0]]
        except:
            shifted_phase = []
        for t in range(len(new_phase) - 1):
            phase_difference = new_phase[t+1] - shifted_phase[t]
            if phase_difference > 180.0:
                shifted_phase.append(new_phase[t+1] - 360.0)
            elif phase_difference < -180.0:
                shifted_phase.append(new_phase[t+1] + 360.0)
            else:
                shifted_phase.append(new_phase[t+1])
        # now comes the fitting to find average phase, phase gradients 
        # and errors
        weights = np.asarray(new_coherence.reshape(
                1, np.shape(new_coherence)[0])[0])
        try:
            # a straight line fit with coherence weighting is used to 
            # determine the phase gradient
            p, V = np.polyfit(new_frequency.reshape(
                       1, np.shape(new_frequency)[0])[0], 
                   shifted_phase, 1, w=weights, cov=True)
            # the average phase is a weighted average
            average_phase[i] = np.average(shifted_phase, 
                         weights=new_coherence)
            phase_gradients[i] = p[0]
            gradient_errors[i] = np.sqrt(V[0][0])
            # find errors on phase as maximum of residuals..
            new_points = p[0]*new_frequency + p[1]
            residuals = shifted_phase - new_points
            phase_errors[i] = np.max(np.abs(residuals))
            frequencies[i] = np.average(new_frequency, 
                       weights=new_coherence)
        except Exception as e:
            try:
                # if there are not enough points for errors on the fitting,
                # leave gradient error as zero
                p = np.polyfit(new_frequency.reshape(
                        1, np.shape(new_frequency)[0])[0], 
                    new_phase, 1, w=weights)
                average_phase[i] = np.average(shifted_phase, 
                             weights=new_coherence)
                phase_gradients[i] = p[0]
                new_points = p[0]*new_frequency + p[1]
                residuals = shifted_phase - new_points
                phase_errors[i] = np.max(residuals)
                frequencies[i] = np.average(new_frequency, 
                           weights=new_coherence)
            except:
                # or just set phase gradient to -1
                phase_gradients[i] = -1.0
                try:
                    average_phase[i] = np.average(shifted_phase, 
                                 weights=new_coherence)
                    frequencies[i] = np.average(new_frequency, 
                               weights=new_coherence)
                except:
                    pass
    results_dict = {
        "timeslice": timeslice, 
        "frequency range": freqrange, 
        "chosen frequency": pickfreq, 
        "numperseg": numperseg, 
        "reference channel": refch, 
        "coherence limit": coherence_limit, 
        "average phase": average_phase, 
        "average phase error": phase_errors, 
        "phase gradient": phase_gradients, 
        "phase gradient error": gradient_errors, 
        "maximum coherence": max_coherence, 
        "average frequencies": frequencies, 
        "phase at chosen frequency": phase_for_specific_freq, 
        "error on phase at chosen frequency": phase_error_for_specific_freq
        }
    return results_dict

# a function to calculate the power in the mode (and another estimate 
# for mode frequency)
def calculate_mode_power(BESdata, timeslice, freqrange, n=7):
    nchan = BESdata.nchan
    start_index = (np.abs(BESdata.time.cut - timeslice[0])).argmin()
    stop_index = (np.abs(BESdata.time.cut - timeslice[1])).argmin()
    bes_dataslice = BESdata.data.fluct[:, start_index:stop_index]

    numperseg = 2**n
    f_bes, Ppf_bes = sig.csd(bes_dataslice, bes_dataslice, fs=BESdata.f_samp, 
                             nperseg=numperseg, window='hann')

    mode_powers = np.full(nchan, np.nan)
    mode_frequencies = np.full(nchan, np.nan)

    for i in range(nchan):    
        # the background is fitted as a straight line in the log-log 
        # domain, neglecting the first 9 entries (mode location)
        logf = np.log(f_bes)
        logP = np.log(Ppf_bes[i])
        p = np.polyfit(logf[9:], logP[9:], 1)
        # ### version 1:
        # # subtract this background from the data 
        # # (leaving the first point as is)
        # subtracted = Ppf_bes[i, 1:] - (f_bes[1:]**p[0])*np.exp(p[1])
        # subtracted = np.concatenate([[Ppf_bes[i, 0]], subtracted])
        # # estimate some first guesses for the Gaussian fit
        # modef = ((freqrange[0] + freqrange[1])/2.0) - 50.0e3
        # modef_width = ((freqrange[1] - freqrange[0])/2.0) - 50.0e3
        # mode_amp = subtracted[(np.abs(f_bes - modef)).argmin()]
        # popt,pcov = curve_fit(gaussian, f_bes, subtracted, 
        #                       p0=[mode_amp, modef, modef_width]) 
        ### version 2:
        newf = np.delete(f_bes, [0,1,2,3,4,7,8])
        newP = np.delete(Ppf_bes[i], [0,1,2,3,4,7,8])
        weights = np.ones_like(newP)
        if freqrange[0]>0.0:
            weights[0] = 3.0
            weights[1] = 3.0
            sumof = newP.shape[0] + 4.0
            weights = weights/sumof
        p_new2 = np.polyfit(np.log(newf), np.log(newP), 1, w=weights)
        # subtract this background from the data 
        # (leaving the first point as is)
        subtracted = Ppf_bes[i, 1:] - (f_bes[1:]**p_new2[0])*np.exp(p_new2[1])
        subtracted = np.concatenate([[Ppf_bes[i, 0]], subtracted])
        # estimate some first guesses for the Gaussian fit
        #modef = ((freqrange[0] + freqrange[1])/2.0) #- 50.0e3
        #modef_width = ((freqrange[1] - freqrange[0])/2.0) - 30.0e3#50.0e3
        #mode_amp = subtracted[(np.abs(f_bes - modef)).argmin()]
        #popt,pcov = curve_fit(gaussian, f_bes[6:18], subtracted[6:18], 
        #                      p0=[mode_amp, modef, modef_width])
        modef = ((freqrange[0] + freqrange[1])/2.0) - 80.0e3
        modef_width = ((freqrange[1] - freqrange[0])/2.0) - 50.0e3
        mode_amp = subtracted[(np.abs(f_bes - modef)).argmin()]
        popt,pcov = curve_fit(gaussian, f_bes, subtracted, 
                              p0=[mode_amp, modef, modef_width])
        ###
        # mode power is integral under Gaussian fit for mode
        integral = np.sqrt(2)*popt[0]*np.abs(popt[2])*np.sqrt(np.pi)
        mode_powers[i] = integral
        # mode frequency only accepted if reasonable
        if popt[1] > 0.0:
            if freqrange[0]<10.0:
                if popt[1]>15.0e3 and popt[1]<90.0e3:
                    mode_frequencies[i] = popt[1]
            else:
                mode_frequencies[i] = popt[1]
    results_dict = {
        "timeslice": timeslice, 
        "frequency range": freqrange, 
        "numperseg": numperseg, 
        "mode powers": mode_powers, 
        "mode frequencies": mode_frequencies
        }
    return results_dict
