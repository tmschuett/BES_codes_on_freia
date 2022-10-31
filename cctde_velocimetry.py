# CCTDE velocimetry function
from imports import *
from BESClass import *
from functions_BES import *

# calculates the poloidal and/or radial velocities for a BES shot and timeslice
# calls functions butter_bandpass_filter, map_channel_to_flux_surfaces, 
# calc_radial_chans_from_equilibrium, cross_corr, gaussian, 
# calc_radial_and_poloidal_with_removed_average
def calc_cctde_velocimetry(
        BESdata, timeslice, xcorr_length, bandpass_filter='automatic', 
        fit_range=0.2, calc_poloidal=True, calc_radial=True, 
        remove_poloidal_average=False, use_gaussian_fit=True, 
        cols_to_calc=[0, 1, 2, 3, 4, 5, 6, 7], rows_to_calc=[0, 1, 2, 3]):
    # options: 
    # - calculate either poloidal or radial or both velocities
    # - can remove the poloidal average before calculating the velocities
    # - choose the cross-correlation length in points (e.g. 200, 500, or 1000)
    # - can use a Gaussian fit for the correlation/time delay calculations 
    #   or just use the maximum. Choose the data range from peak for the 
    #   Gaussian fit with fit_range (as fraction of xcorr_length)
    # - choose which column and rows to calculate (e.g. for testing, can run 
    #   for one channel only to speed up)
    # - can choose how/whether to bandpass filter the BES data beforehand, 
    #   ('automatic', None, or list of [lower, upper])
    
    shot = BESdata.shot_number
    nrows = BESdata.nrows
    ncols = BESdata.ncols
    f_samp = BESdata.f_samp
    apdpos = BESdata.apdpos
    bes_time = BESdata.time.cut
    
    nchan = nrows*ncols
    
    idx1 = (np.abs(bes_time - timeslice[0])).argmin()
    idx2 = (np.abs(bes_time - timeslice[1])).argmin()
    timeslice_len = idx2 - idx1
    
    # try sliding by 1 time point
    slide_point = 1
    num_slices = int((timeslice_len - xcorr_length) / slide_point) + 1
    print("Timeslice length: " + str(timeslice_len) + 
          ", with cross-correlation length: " + str(xcorr_length) + 
          ", resulting in " + str(num_slices) + " slices.")
    
    # apply bandpass filter if requested
    bpf_filt_data = []
    if bandpass_filter == 'automatic':
        for j in range(nchan):
            bpf_filt = butter_bandpass_filter(BESdata.data.fluct[j], 
                                              (0.5 * f_samp) / xcorr_length, 
                                              250.0e3, f_samp)
            bpf_filt_data.append(bpf_filt)
        print("Applied automatic bandpass filter.")
    elif bandpass_filter == None:
        for j in range(nchan):
            bpf_filt_data.append(BESdata.data.fluct[j])
        print("No additional filtering.")
    else:
        try:
            for j in range(nchan):
                bpf_filt = butter_bandpass_filter(BESdata.data.fluct[j], 
                                                  bandpass_filter[0], 
                                                  bandpass_filter[1], 
                                                  f_samp)
                bpf_filt_data.append(bpf_filt)
            print("Applied custom bandpass filter.")
        except:
            print('bandpass_filter must be "automatic", None, or '
                  'a list of [lower, upper]. Using "automatic".')
            for j in range(nchan):
                bpf_filt = butter_bandpass_filter(
                    BESdata.data.fluct[j], (0.5 * f_samp) / xcorr_length, 
                    250.0e3, f_samp)
                bpf_filt_data.append(bpf_filt)
            print("Applied automatic bandpass filter.")
    
    bpf_data = np.asarray(bpf_filt_data)
    
    # get the list of EFIT times
    psi = udaClient.get('efm_psi(r,z)', shot)
    eq_idx1 = (np.abs(psi.time.data - timeslice[0])).argmin()
    eq_idx2 = (np.abs(psi.time.data - timeslice[1])).argmin()
    # only calculate equilibria for given timeslice
    cut_eq_times = psi.time.data[eq_idx1-1:eq_idx2+2]
    
    results_dict = {}
    results_dict["removed average"] = remove_poloidal_average
    results_dict["use Gaussian fit"] = use_gaussian_fit
    
    if remove_poloidal_average is True:
        print("Removing poloidal average...")
        # EFIT equilibrium has fairly low time resolution, so it's more 
        # efficient to perform these calculations at the start for each 
        # equilibrium timepoint rather than calling the function for each BES 
        # timepoint. 
        eq_array_list = []
        for eq_idx in range(len(cut_eq_times)):
            eq_timepoint = cut_eq_times[eq_idx]
            eq_array = np.zeros((nrows, ncols))
            for row in range(nrows):
                for col in range(ncols):
                    # calculate the equilibrium value for each BES channel 
                    # at that timepoint
                    eq_array[row, col] = map_channel_to_flux_surfaces(shot, 
                                eq_timepoint, apdpos, ncols * row + col)
            eq_array_list.append(eq_array)
        print("Equilibrium values for each channel calculated.")
        
        if calc_radial is True:
            eq_rad_chans_list = []
            for eq_idx in range(len(cut_eq_times)):
                # make an array of chosen radial channels for each reference 
                # BES channel based on equilibrium
                eq_timepoint = cut_eq_times[eq_idx]
                equilib = eq.equilibrium(device='MAST', shot=shot, 
                                         time=eq_timepoint)
                eq_rad_chans = calc_radial_chans_from_equilibrium(
                    equilib.R, equilib.Z, np.sqrt(equilib.psiN), apdpos, BESdata)
                eq_rad_chans_list.append(eq_rad_chans)
            print("Radial channels chosen based on equilibrium values.")
            
            if calc_poloidal is True:
                # if calculating both radial and poloidal for removed average, 
                # call function (slightly more efficiently written)
                print("Calculating poloidal and radial velocities...")
                temp_results_dict = calc_radial_and_poloidal_with_removed_average(
                    bpf_data, cut_eq_times, apdpos, use_gaussian_fit, 
                    num_slices, slide_point, xcorr_length, idx1, bes_time, 
                    eq_array_list, eq_rad_chans_list, ncols, nrows, f_samp, 
                    cols_to_calc, rows_to_calc, fit_range)
                print("Finished calculating velocities.")
                results_dict["poloidal"] = temp_results_dict["poloidal"]
                results_dict["radial"] = temp_results_dict["radial"]
            else:
                # if only radial for removed average
                results_dict["radial"] = {}
                
                mid_time_points = []
                radial_velocities = []
                radial_correlation = []
                radial_errors = []
                
                print("Calculating radial velocities...")
                new_fluct_data = np.copy(bpf_data)
                
                if use_gaussian_fit is True:
                    for j in range(num_slices):
                        # indices for the current slice over which to correlate
                        index1 = idx1 + j * slide_point
                        index2 = index1 + xcorr_length
                        length_chunk = (index2 - index1)
                        
                        # the mid-point of the slice - take this to be the 
                        # time point for the velocity
                        mid_time_point = 0.5 * (bes_time[index1] + 
                                                bes_time[index2])
                        
                        # for removing poloidal average, find poloidal 
                        # channels for each reference channel
                        
                        # choose the equilibrium value array for the timepoint
                        eq_list_idx = (np.abs(cut_eq_times - 
                                              mid_time_point)).argmin()
                        eq_array_to_use = eq_array_list[eq_list_idx]
                        for i_x in range(ncols):
                            for k_x in range(nrows):
                                refch_x = i_x + ncols * k_x
                                # find the poloidal channels for each channel
                                refch_eq = eq_array_to_use[k_x, i_x]
                                chans_pol = []
                                for row_x in range(nrows):
                                    col_x = (np.abs(
                                        eq_array_to_use[row_x, :] - 
                                        refch_eq)).argmin()
                                    chans_pol.append(col_x + ncols * row_x)
                                
                                # find the poloidal average
                                average = (bpf_data[chans_pol[0], :] + 
                                           bpf_data[chans_pol[1], :] + 
                                           bpf_data[chans_pol[2], :] + 
                                           bpf_data[chans_pol[3], :]) / 4.0
                                # subtract poloidal average from the data
                                new_fluct_data[refch_x, :] = bpf_data[
                                    refch_x, :] - average
                        
                        # get array of chosen radial channels for each 
                        # reference BES channel based on equilibrium
                        eq_rad_chans_to_use = eq_rad_chans_list[eq_list_idx]
                        
                        # the time base for the time delay estimation
                        time_base = -(1.0 / f_samp) * np.arange(
                            -int(length_chunk / 2), int(length_chunk / 2), 1)
                        
                        # the radial velocity array for this time slice
                        rad_vels = np.zeros((nrows, ncols))
                        # the associated average correlation values
                        rad_corr = np.zeros((nrows, ncols))
                        # the error on the gradient fit (i.e. the velocity)
                        rad_err = np.zeros((nrows, ncols))
                        
                        # goes through all columns and rows unless otherwise 
                        # specified
                        for i in cols_to_calc:
                            for k in rows_to_calc:
                                # the reference channel:
                                refch = i + ncols * k
                                # the equilibrium value for the reference 
                                # channel
                                refch_eq = eq_array_to_use[k, i]
                                
                                # the distances, time delays and cross 
                                # correlation values 
                                # of the peaks
                                distances = []
                                time_dels = []
                                weights = []
                                
                                # this sets the calculation to use only the nearest 
                                # neighbour channels.. do I want to add an option to use 
                                # more channels? or otherwise fix the previous calculation 
                                # of chosen channels to only consider nearest neighbour 
                                # columns..
                                errors = True
                                chans_1 = []
                                for ccc in range(ncols):
                                    if eq_rad_chans_to_use[refch, ccc] > -1:
                                        chans_1.append(
                                            eq_rad_chans_to_use[refch, ccc])
                                rc_idx = chans_1.index(refch)
                                if rc_idx > 0:
                                    if rc_idx < len(chans_1) - 1:
                                        chans = [chans_1[rc_idx - 1], 
                                                 chans_1[rc_idx], 
                                                 chans_1[rc_idx + 1]]
                                    else:
                                        if rc_idx > 1:
                                            chans = [chans_1[rc_idx - 2], 
                                                     chans_1[rc_idx - 1], 
                                                     chans_1[rc_idx]]
                                        else:
                                            chans = [chans_1[rc_idx - 1], 
                                                     chans_1[rc_idx]]
                                else:
                                    if len(chans_1) > 2:
                                        chans = [chans_1[rc_idx], 
                                                 chans_1[rc_idx + 1], 
                                                 chans_1[rc_idx + 2]]
                                    else:
                                        chans = chans_1
                                if len(chans) < 3:
                                    errors = False
                                
                                # go through the four chosen channels
                                for ch in chans:
                                    # the cross-correlation for this channel with the 
                                    # reference channel
                                    xcorr = cross_corr(
                                        refch, ch, new_fluct_data, index1, 
                                        index2)
                                    # scipy.optimize.curve_fit on freia doesn't like the 
                                    # complex data, so convert to float
                                    xcorr = xcorr.astype(float)
                                    
                                    # the distance between channels
                                    if ch == refch:
                                        distances.append(0.0)
                                    else:
                                        # includes sign for direction
                                        distances.append(
                                            ((ch - refch) / np.abs(ch - refch)) * 
                                            np.sqrt(
                                                np.abs(apdpos[ch][1] - 
                                                       apdpos[refch][1]) ** 2 + 
                                                np.abs(apdpos[ch][0] - 
                                                       apdpos[refch][0]) ** 2))
                                    # the time delay of the peak 
                                    # (maximum correlation value)
                                    time_del_peak = time_base[np.argmax(xcorr)]
                                    # the value of the cross-correlation at peak
                                    maxcorr_peak = np.max(xcorr)
                                    
                                    # the time base index of the peak
                                    peak_idx = np.argwhere(time_base == 
                                                           time_del_peak)[0][0]
                                    
                                    # find the start and end indices of the slice over 
                                    # which to perform a Gaussian fit, based on the 
                                    # fit_range parameter
                                    idx_stop = peak_idx - int(
                                        (fit_range / 2) * length_chunk)
                                    if idx_stop < 0:
                                        idx_stop = 0
                                    
                                    idx_start = peak_idx + int(
                                        (fit_range / 2) * length_chunk)
                                    if idx_start > len(time_base) - 1:
                                        idx_start = len(time_base) - 1
                                    
                                    # need to flip the time base and the correlation values
                                    rev_time = np.flip(time_base[idx_stop:idx_start])
                                    rev_corr = np.flip(xcorr[idx_stop:idx_start])
                                    
                                    try:
                                        # try to fit a Gaussian
                                        popt, pcov = curve_fit(
                                            gaussian, rev_time, rev_corr, p0=[
                                                maxcorr_peak, time_del_peak, 
                                                (rev_time[-1] - rev_time[0]) / 2.0])
                                        # choose new peak time and correlation, based on 
                                        # Gaussian fit
                                        time_val = rev_time[np.argmax(gaussian(
                                            rev_time, *popt))]
                                        corr_val = np.max(gaussian(rev_time, *popt))
                                    except:
                                        # if fit does not work, choose peak based on 
                                        # maxixmum correlation
                                        time_val = time_del_peak
                                        corr_val = maxcorr_peak
                                    time_dels.append(time_val)
                                    # assign the correlation values as weights
                                    weights.append(corr_val)
                                
                                # calculate the average correlation
                                av_corr = (np.sum(np.asarray(weights))) / len(chans)
                                rad_corr[k][i] = av_corr
                                try:
                                    # try to fit a weighted linear fit to the distances 
                                    # and time delays (weighted by correlation)
                                    p = np.polyfit(distances, time_dels, 1, 
                                                   w=np.asarray(weights))
                                    # velocity is 1/slope
                                    rad_vels[k][i] = 1.0 / p[0]
                                except:
                                    try:
                                        # if it doesn't work, try unweighted fit
                                        p = np.polyfit(distances, time_dels, 1)
                                        rad_vels[k][i] = 1.0 / p[0]
                                    except:
                                        # if no fit is possible (e.g. not enough channels)
                                        rad_vels[k][i] = np.NaN
                                
                                if errors:
                                    # for error estimation, interpolate distance, time delay 
                                    # and correlation arrays
                                    dist_new = np.linspace(distances[0], distances[-1], 9)
                                    time_interp = interp1d(distances, time_dels)
                                    weight_interp = interp1d(distances, weights)
                                    time_new = time_interp(dist_new)
                                    weights_new = weight_interp(dist_new)
                                    try:
                                        # try to fit interpolated array with covariance to 
                                        # get error on slope
                                        p,V = np.polyfit(dist_new, time_new, 1, 
                                                       w=np.asarray(weights_new), cov=True)
                                        rad_err[k][i] = np.sqrt(V[0][0])
                                    except:
                                        # if it doesn't work, set error as NaN
                                        rad_err[k][i] = np.NaN
                                else:
                                    rad_err[k][i] = np.NaN
                                
                        # append arrays of values for all channels (for one timeslice)
                        mid_time_points.append(mid_time_point)
                        radial_velocities.append(rad_vels)
                        radial_correlation.append(rad_corr)
                        radial_errors.append(rad_err)
                else:
                    for j in range(num_slices):
                        # indices for the current slice over which to correlate
                        index1 = idx1 + j * slide_point
                        index2 = index1 + xcorr_length
                        length_chunk = (index2 - index1)
                        
                        # the mid-point of the slice - take this to be the time point for the velocity
                        mid_time_point = 0.5 * (bes_time[index1] + bes_time[index2])
                        
                        # for removing poloidal average, find poloidal channels for each 
                        # reference channel
                        
                        # choose the equilibrium value array for the timepoint
                        eq_list_idx = (np.abs(cut_eq_times - mid_time_point)).argmin()
                        eq_array_to_use = eq_array_list[eq_list_idx]
                        for i_x in range(ncols):
                            for k_x in range(nrows):
                                refch_x = i_x + ncols * k_x
                                # find the poloidal channels for each channel
                                refch_eq = eq_array_to_use[k_x, i_x]
                                chans_pol = []
                                for row_x in range(nrows):
                                    col_x = (np.abs(eq_array_to_use[row_x, :] - 
                                                    refch_eq)).argmin()
                                    chans_pol.append(col_x + ncols * row_x)
                                
                                # find the poloidal average
                                average = (bpf_data[chans_pol[0], :] + 
                                           bpf_data[chans_pol[1], :] + 
                                           bpf_data[chans_pol[2], :] + 
                                           bpf_data[chans_pol[3], :]) / 4.0
                                # subtract poloidal average from the data
                                new_fluct_data[refch_x, :] = bpf_data[refch_x, :] - average
                        
                        # get array of chosen radial channels for each reference BES 
                        # channel based on equilibrium
                        eq_rad_chans_to_use = eq_rad_chans_list[eq_list_idx]
                        
                        # the time base for the time delay estimation
                        time_base = -(1.0 / f_samp) * np.arange(-int(length_chunk / 2), 
                                      int(length_chunk / 2), 1)
                        
                        # the radial velocity array for this time slice
                        rad_vels = np.zeros((nrows, ncols))
                        # the associated average correlation values
                        rad_corr = np.zeros((nrows, ncols))
                        # the error on the gradient fit (i.e. the velocity)
                        rad_err = np.zeros((nrows, ncols))
                        
                        # goes through all columns and rows unless otherwise specified
                        for i in cols_to_calc:
                            for k in rows_to_calc:
                                # the reference channel:
                                refch = i + ncols * k
                                # the equilibrium value for the reference channel
                                refch_eq = eq_array[k, i]
                                
                                # the distances, time delays and cross correlation values 
                                # of the peaks
                                distances = []
                                time_dels = []
                                weights = []
                                
                                # this sets the calculation to use only the nearest 
                                # neighbour channels.. do I want to add an option to use 
                                # more channels? or otherwise fix the previous calculation 
                                # of chosen channels to only consider nearest neighbour 
                                # columns..
                                errors = True
                                chans_1 = []
                                for ccc in range(ncols):
                                    if eq_rad_chans_to_use[refch, ccc] > -1:
                                        chans_1.append(eq_rad_chans_to_use[refch, ccc])
                                rc_idx = chans_1.index(refch)
                                if rc_idx > 0:
                                    if rc_idx < len(chans_1) - 1:
                                        chans = [chans_1[rc_idx - 1], 
                                                 chans_1[rc_idx], 
                                                 chans_1[rc_idx + 1]]
                                    else:
                                        if rc_idx > 1:
                                            chans = [chans_1[rc_idx - 2], 
                                                     chans_1[rc_idx - 1], 
                                                     chans_1[rc_idx]]
                                        else:
                                            chans = [chans_1[rc_idx - 1], 
                                                     chans_1[rc_idx]]
                                else:
                                    if len(chans_1) > 2:
                                        chans = [chans_1[rc_idx], 
                                                 chans_1[rc_idx + 1], 
                                                 chans_1[rc_idx + 2]]
                                    else:
                                        chans = chans_1
                                if len(chans) < 3:
                                    errors = False
                                
                                # go through the four chosen channels
                                for ch in chans:
                                    # the cross-correlation for this channel with the 
                                    # reference channel
                                    xcorr = cross_corr(refch, ch, new_fluct_data, index1, 
                                                       index2)
                                    # scipy.optimize.curve_fit on freia doesn't like the 
                                    # complex data, so convert to float
                                    xcorr = xcorr.astype(float)
                                    
                                    # the distance between channels
                                    if ch == refch:
                                        distances.append(0.0)
                                    else:
                                        # includes sign for direction
                                        distances.append(
                                            ((ch - refch) / np.abs(ch - refch)) * 
                                            np.sqrt(
                                                np.abs(apdpos[ch][1] - 
                                                       apdpos[refch][1]) ** 2 + 
                                                np.abs(apdpos[ch][0] - 
                                                       apdpos[refch][0]) ** 2))
                                    # the time delay of the peak 
                                    # (maximum correlation value)
                                    time_del_peak = time_base[np.argmax(xcorr)]
                                    # the value of the cross-correlation at peak
                                    maxcorr_peak = np.max(xcorr)
                                    # choose peak based on maxixmum correlation
                                    time_dels.append(time_del_peak)
                                    # assign the correlation values as weights
                                    weights.append(maxcorr_peak)
                                
                                # calculate the average correlation
                                av_corr = (np.sum(np.asarray(weights))) / len(chans)
                                rad_corr[k][i] = av_corr
                                try:
                                    # try to fit a weighted linear fit to the distances 
                                    # and time delays (weighted by correlation)
                                    p = np.polyfit(distances, time_dels, 1, 
                                                   w=np.asarray(weights))
                                    # velocity is 1/slope
                                    rad_vels[k][i] = 1.0 / p[0]
                                except:
                                    try:
                                        # if it doesn't work, try unweighted fit
                                        p = np.polyfit(distances, time_dels, 1)
                                        rad_vels[k][i] = 1.0 / p[0]
                                    except:
                                        # if no fit is possible (e.g. not enough channels)
                                        rad_vels[k][i] = np.NaN
                                
                                if errors:
                                    # for error estimation, interpolate distance, time delay 
                                    # and correlation arrays
                                    dist_new = np.linspace(distances[0], distances[-1], 9)
                                    time_interp = interp1d(distances, time_dels)
                                    weight_interp = interp1d(distances, weights)
                                    time_new = time_interp(dist_new)
                                    weights_new = weight_interp(dist_new)
                                    try:
                                        # try to fit interpolated array with covariance to 
                                        # get error on slope
                                        p,V = np.polyfit(dist_new, time_new, 1, 
                                                       w=np.asarray(weights_new), cov=True)
                                        rad_err[k][i] = np.sqrt(V[0][0])
                                    except:
                                        # if it doesn't work, set error as NaN
                                        rad_err[k][i] = np.NaN
                                else:
                                    rad_err[k][i] = np.NaN
                                
                        # append arrays of values for all channels (for one timeslice)
                        mid_time_points.append(mid_time_point)
                        radial_velocities.append(rad_vels)
                        radial_correlation.append(rad_corr)
                        radial_errors.append(rad_err)
                print("Finished radial velocity calculations.")
                results_dict["radial"]["mid_time_points"] = mid_time_points
                results_dict["radial"]["velocities"] = radial_velocities
                results_dict["radial"]["correlations"] = radial_correlation
                results_dict["radial"]["errors"] = radial_errors
        elif calc_poloidal is True:
            # if only poloidal for removed average
            print("Calculating poloidal velocities...")
            results_dict["poloidal"] = {}
            
            mid_time_points = []
            poloidal_velocities = []
            poloidal_correlation = []
            poloidal_errors = []
            new_fluct_data = np.copy(bpf_data)
            
            if use_gaussian_fit is True:
                for j in range(num_slices):
                    # indices for the current slice over which to correlate
                    index1 = idx1 + j * slide_point
                    index2 = index1 + xcorr_length
                    length_chunk = (index2 - index1)
                    
                    # the mid-point of the slice - take this to be the time point for the velocity
                    mid_time_point = 0.5 * (bes_time[index1] + bes_time[index2])
                    
                    # choose the equilibrium value array for the timepoint
                    eq_list_idx = (np.abs(cut_eq_times - mid_time_point)).argmin()
                    eq_array_to_use = eq_array_list[eq_list_idx]
                    
                    # the time base for the time delay estimation
                    time_base = -(1.0 / f_samp) * np.arange(-int(length_chunk / 2), 
                                  int(length_chunk / 2), 1)
                    
                    # the poloidal velocity array for this time slice
                    pol_vels = np.zeros((nrows, ncols))
                    # the associated average correlation values
                    pol_corr = np.zeros((nrows, ncols))
                    # the error on the gradient fit (i.e. the velocity)
                    pol_err = np.zeros((nrows, ncols))
                    
                    # goes through all columns and rows unless otherwise specified
                    for i in cols_to_calc:
                        for k in rows_to_calc:
                            # the reference channel:
                            refch = i + ncols * k
                            # the equilibrium value for the reference channel
                            refch_eq = eq_array_to_use[k, i]
                            # finding four channels to correlate with (including itself)
                            # based on which channel in each row has the closest 
                            # equilibrium value
                            chans = []
                            for row_x in range(nrows):
                                col_x = (np.abs(eq_array_to_use[row_x, :] - 
                                                refch_eq)).argmin()
                                chans.append(col_x + ncols * row_x)
                            
                            # find the poloidal average
                            average = (bpf_data[chans[0], :] + 
                                       bpf_data[chans[1], :] + 
                                       bpf_data[chans[2], :] + 
                                       bpf_data[chans[3], :]) / 4.0
                            # subtract poloidal average from the data for the four 
                            # channels considered
                            for new_chan in chans:
                                new_fluct_data[new_chan, :] = (
                                        bpf_data[new_chan, :] - average)
                            
                            # the distances, time delays and cross correlation values 
                            # of the peaks
                            distances = []
                            time_dels = []
                            weights = []
                            
                            # go through the four chosen channels
                            for ch in chans:
                                # the cross-correlation for this channel with the 
                                # reference channel
                                xcorr = cross_corr(refch, ch, new_fluct_data, index1, 
                                                   index2)
                                # scipy.optimize.curve_fit on freia doesn't like the 
                                # complex data, so convert to float
                                xcorr = xcorr.astype(float)
                                
                                # the distance between channels
                                if ch == refch:
                                    distances.append(0.0)
                                else:
                                    # includes sign for direction
                                    distances.append(
                                        ((ch - refch) / np.abs(ch - refch)) * 
                                        np.sqrt(
                                            np.abs(apdpos[ch][1] - 
                                                   apdpos[refch][1]) ** 2 + 
                                            np.abs(apdpos[ch][0] - 
                                                   apdpos[refch][0]) ** 2))
                                # the time delay of the peak 
                                # (maximum correlation value)
                                time_del_peak = time_base[np.argmax(xcorr)]
                                # the value of the cross-correlation at peak
                                maxcorr_peak = np.max(xcorr)
                                
                                # the time base index of the peak
                                peak_idx = np.argwhere(time_base == 
                                                       time_del_peak)[0][0]
                                
                                # find the start and end indices of the slice over 
                                # which to perform a Gaussian fit, based on the 
                                # fit_range parameter
                                idx_stop = peak_idx - int((fit_range / 2) * length_chunk)
                                if idx_stop < 0:
                                    idx_stop = 0
                                
                                idx_start = peak_idx + int((fit_range / 2) * length_chunk)
                                if idx_start > len(time_base) - 1:
                                    idx_start = len(time_base) - 1
                                
                                # need to flip the time base and the correlation values
                                rev_time = np.flip(time_base[idx_stop:idx_start])
                                rev_corr = np.flip(xcorr[idx_stop:idx_start])
                                
                                try:
                                    # try to fit a Gaussian
                                    popt, pcov = curve_fit(
                                        gaussian, rev_time, rev_corr, p0=[
                                            maxcorr_peak, time_del_peak, 
                                            (rev_time[-1] - rev_time[0]) / 2.0])
                                    # choose new peak time and correlation, based on 
                                    # Gaussian fit
                                    time_val = rev_time[np.argmax(gaussian(
                                        rev_time, *popt))]
                                    corr_val = np.max(gaussian(rev_time, *popt))
                                except:
                                    # if fit does not work, choose peak based on 
                                    # maxixmum correlation
                                    time_val = time_del_peak
                                    corr_val = maxcorr_peak
                                time_dels.append(time_val)
                                # assign the correlation values as weights
                                weights.append(corr_val)
                            
                            # calculate the average correlation
                            av_corr = (np.sum(np.asarray(weights))) / 4.0
                            pol_corr[k][i] = av_corr
                            try:
                                # try to fit a weighted linear fit to the distances 
                                # and time delays (weighted by correlation), 
                                # with slope = velocity
                                p = np.polyfit(distances, time_dels, 1, 
                                               w=np.asarray(weights))
                            except:
                                # if it doesn't work, try unweighted fit
                                p = np.polyfit(distances, time_dels, 1)
                            
                            # for error estimation, interpolate distance, time delay 
                            # and correlation arrays
                            dist_new = np.linspace(distances[0], distances[-1], 9)
                            time_interp = interp1d(distances, time_dels)
                            weight_interp = interp1d(distances, weights)
                            time_new = time_interp(dist_new)
                            weights_new = weight_interp(dist_new)
                            try:
                                # try to fit interpolated array with covariance to 
                                # get error on slope
                                p,V = np.polyfit(dist_new, time_new, 1, 
                                               w=np.asarray(weights_new), cov=True)
                                pol_err[k][i] = np.sqrt(V[0][0])
                            except:
                                # if it doesn't work, set error as NaN
                                pol_err[k][i] = np.NaN
                            
                            # velocity is 1/slope
                            pol_vels[k][i] = 1.0 / p[0]
                    # append arrays of values for all channels (for one timeslice)
                    mid_time_points.append(mid_time_point)
                    poloidal_velocities.append(pol_vels)
                    poloidal_correlation.append(pol_corr)
                    poloidal_errors.append(pol_err)
            else:
                for j in range(num_slices):
                    # indices for the current slice over which to correlate
                    index1 = idx1 + j * slide_point
                    index2 = index1 + xcorr_length
                    length_chunk = (index2 - index1)
                    
                    # the mid-point of the slice - take this to be the time point for the velocity
                    mid_time_point = 0.5 * (bes_time[index1] + bes_time[index2])
                    
                    # choose the equilibrium value array for the timepoint
                    eq_list_idx = (np.abs(cut_eq_times - mid_time_point)).argmin()
                    eq_array_to_use = eq_array_list[eq_list_idx]
                    
                    # the time base for the time delay estimation
                    time_base = -(1.0 / f_samp) * np.arange(-int(length_chunk / 2), 
                                  int(length_chunk / 2), 1)
                    
                    # the poloidal velocity array for this time slice
                    pol_vels = np.zeros((nrows, ncols))
                    # the associated average correlation values
                    pol_corr = np.zeros((nrows, ncols))
                    # the error on the gradient fit (i.e. the velocity)
                    pol_err = np.zeros((nrows, ncols))
                    
                    # goes through all columns and rows unless otherwise specified
                    for i in cols_to_calc:
                        for k in rows_to_calc:
                            # the reference channel:
                            refch = i + ncols * k
                            # the equilibrium value for the reference channel
                            refch_eq = eq_array_to_use[k, i]
                            # finding four channels to correlate with (including itself)
                            # based on which channel in each row has the closest 
                            # equilibrium value
                            chans = []
                            for row_x in range(nrows):
                                col_x = (np.abs(eq_array_to_use[row_x, :] - 
                                                refch_eq)).argmin()
                                chans.append(col_x + ncols * row_x)
                            
                            # find the poloidal average
                            average = (bpf_data[chans[0], :] + 
                                       bpf_data[chans[1], :] + 
                                       bpf_data[chans[2], :] + 
                                       bpf_data[chans[3], :]) / 4.0
                            # subtract poloidal average from the data for the four 
                            # channels considered
                            for new_chan in chans:
                                new_fluct_data[new_chan, :] = (
                                        bpf_data[new_chan, :] - average)
                            
                            # the distances, time delays and cross correlation values 
                            # of the peaks
                            distances = []
                            time_dels = []
                            weights = []
                            
                            # go through the four chosen channels
                            for ch in chans:
                                # the cross-correlation for this channel with the 
                                # reference channel
                                xcorr = cross_corr(refch, ch, new_fluct_data, index1, 
                                                   index2)
                                # scipy.optimize.curve_fit on freia doesn't like the 
                                # complex data, so convert to float
                                xcorr = xcorr.astype(float)
                                
                                # the distance between channels
                                if ch == refch:
                                    distances.append(0.0)
                                else:
                                    # includes sign for direction
                                    distances.append(
                                        ((ch - refch) / np.abs(ch - refch)) * 
                                        np.sqrt(
                                            np.abs(apdpos[ch][1] - 
                                                   apdpos[refch][1]) ** 2 + 
                                            np.abs(apdpos[ch][0] - 
                                                   apdpos[refch][0]) ** 2))
                                # the time delay of the peak 
                                # (maximum correlation value)
                                time_del_peak = time_base[np.argmax(xcorr)]
                                # the value of the cross-correlation at peak
                                maxcorr_peak = np.max(xcorr)
                                # choose peak based on maxixmum correlation
                                time_dels.append(time_del_peak)
                                # assign the correlation values as weights
                                weights.append(maxcorr_peak)
                            
                            # calculate the average correlation
                            av_corr = (np.sum(np.asarray(weights))) / 4.0
                            pol_corr[k][i] = av_corr
                            try:
                                # try to fit a weighted linear fit to the distances 
                                # and time delays (weighted by correlation), 
                                # with slope = velocity
                                p = np.polyfit(distances, time_dels, 1, 
                                               w=np.asarray(weights))
                            except:
                                # if it doesn't work, try unweighted fit
                                p = np.polyfit(distances, time_dels, 1)
                            
                            # for error estimation, interpolate distance, time delay 
                            # and correlation arrays
                            dist_new = np.linspace(distances[0], distances[-1], 9)
                            time_interp = interp1d(distances, time_dels)
                            weight_interp = interp1d(distances, weights)
                            time_new = time_interp(dist_new)
                            weights_new = weight_interp(dist_new)
                            try:
                                # try to fit interpolated array with covariance to 
                                # get error on slope
                                p,V = np.polyfit(dist_new, time_new, 1, 
                                               w=np.asarray(weights_new), cov=True)
                                pol_err[k][i] = np.sqrt(V[0][0])
                            except:
                                # if it doesn't work, set error as NaN
                                pol_err[k][i] = np.NaN
                            
                            # velocity is 1/slope
                            pol_vels[k][i] = 1.0 / p[0]
                    # append arrays of values for all channels (for one timeslice)
                    mid_time_points.append(mid_time_point)
                    poloidal_velocities.append(pol_vels)
                    poloidal_correlation.append(pol_corr)
                    poloidal_errors.append(pol_err)
            
            print("Finished poloidal velocity calculations.")
            results_dict["poloidal"]["mid_time_points"] = mid_time_points
            results_dict["poloidal"]["velocities"] = poloidal_velocities
            results_dict["poloidal"]["correlations"] = poloidal_correlation
            results_dict["poloidal"]["errors"] = poloidal_errors
    else:
        if calc_poloidal is True:
            # EFIT equilibrium has fairly low time resolution, so it's more efficient 
            # to perform these calculations at the start for each equilibrium timepoint 
            # rather than calling the function for each BES timepoint. 
            eq_array_list = []
            for eq_idx in range(len(cut_eq_times)):
                eq_timepoint = cut_eq_times[eq_idx]
                eq_array = np.zeros((nrows, ncols))
                for row in range(nrows):
                    for col in range(ncols):
                        # calculate the equilibrium value for each BES channel 
                        # at that timepoint
                        eq_array[row, col] = map_channel_to_flux_surfaces(shot, 
                                    eq_timepoint, apdpos, ncols * row + col)
                eq_array_list.append(eq_array)
            print("Equilibrium values for each channel calculated.")
            print("Calculating poloidal velocities...")
            results_dict["poloidal"] = {}
            
            mid_time_points = []
            poloidal_velocities = []
            poloidal_correlation = []
            poloidal_errors = []
            
            if use_gaussian_fit is True:
                for j in range(num_slices):
                    # indices for the current slice over which to correlate
                    index1 = idx1 + j * slide_point
                    index2 = index1 + xcorr_length
                    length_chunk = (index2 - index1)
                    
                    # the mid-point of the slice - take this to be the time point for the velocity
                    mid_time_point = 0.5 * (bes_time[index1] + bes_time[index2])
                    
                    # choose the equilibrium value array for the timepoint
                    eq_list_idx = (np.abs(cut_eq_times - mid_time_point)).argmin()
                    eq_array_to_use = eq_array_list[eq_list_idx]
                    
                    # the time base for the time delay estimation
                    time_base = -(1.0 / f_samp) * np.arange(-int(length_chunk / 2), 
                                  int(length_chunk / 2), 1)
                    
                    # the poloidal velocity array for this time slice
                    pol_vels = np.zeros((nrows, ncols))
                    # the associated average correlation values
                    pol_corr = np.zeros((nrows, ncols))
                    # the error on the gradient fit (i.e. the velocity)
                    pol_err = np.zeros((nrows, ncols))
                    
                    # goes through all columns and rows unless otherwise specified
                    for i in cols_to_calc:
                        for k in rows_to_calc:
                            # the reference channel:
                            refch = i + ncols * k
                            # the equilibrium value for the reference channel
                            refch_eq = eq_array_to_use[k, i]
                            # finding four channels to correlate with (including itself)
                            # based on which channel in each row has the closest 
                            # equilibrium value
                            chans = []
                            for row_x in range(nrows):
                                col_x = (np.abs(eq_array_to_use[row_x, :] - 
                                                refch_eq)).argmin()
                                chans.append(col_x + ncols * row_x)
                            
                            # the distances, time delays and cross correlation values 
                            # of the peaks
                            distances = []
                            time_dels = []
                            weights = []
                            
                            # go through the four chosen channels
                            for ch in chans:
                                # the cross-correlation for this channel with the 
                                # reference channel
                                xcorr = cross_corr(refch, ch, bpf_data, index1, 
                                                   index2)
                                # scipy.optimize.curve_fit on freia doesn't like the 
                                # complex data, so convert to float
                                xcorr = xcorr.astype(float)
                                
                                # the distance between channels
                                if ch == refch:
                                    distances.append(0.0)
                                else:
                                    # includes sign for direction
                                    distances.append(
                                        ((ch - refch) / np.abs(ch - refch)) * np.sqrt(
                                            np.abs(apdpos[ch][1] - 
                                                   apdpos[refch][1]) ** 2 + 
                                            np.abs(apdpos[ch][0] - 
                                                   apdpos[refch][0]) ** 2))
                                # the time delay of the peak 
                                # (maximum correlation value)
                                time_del_peak = time_base[np.argmax(xcorr)]
                                # the value of the cross-correlation at peak
                                maxcorr_peak = np.max(xcorr)
                                
                                # the time base index of the peak
                                peak_idx = np.argwhere(time_base == 
                                                       time_del_peak)[0][0]
                                
                                # find the start and end indices of the slice over 
                                # which to perform a Gaussian fit, based on the 
                                # fit_range parameter
                                idx_stop = peak_idx - int((fit_range / 2) * length_chunk)
                                if idx_stop < 0:
                                    idx_stop = 0
                                
                                idx_start = peak_idx + int((fit_range / 2) * length_chunk)
                                if idx_start > len(time_base) - 1:
                                    idx_start = len(time_base) - 1
                                
                                # need to flip the time base and the correlation values
                                rev_time = np.flip(time_base[idx_stop:idx_start])
                                rev_corr = np.flip(xcorr[idx_stop:idx_start])
                                
                                try:
                                    # try to fit a Gaussian
                                    
                                    popt, pcov = curve_fit(
                                        gaussian, rev_time, rev_corr, p0=[
                                            maxcorr_peak, time_del_peak, 
                                            (rev_time[-1] - rev_time[0]) / 2.0])
                                    # choose new peak time and correlation, based on 
                                    # Gaussian fit
                                    time_val = rev_time[np.argmax(gaussian(
                                        rev_time, *popt))]
                                    corr_val = np.max(gaussian(rev_time, *popt))
                                except:
                                    # if fit does not work, choose peak based on 
                                    # maxixmum correlation
                                    time_val = time_del_peak
                                    corr_val = maxcorr_peak
                                time_dels.append(time_val)
                                # assign the correlation values as weights
                                weights.append(corr_val)
                            
                            # calculate the average correlation
                            av_corr = (np.sum(np.asarray(weights))) / (len(chans))
                            pol_corr[k][i] = av_corr
                            try:
                                # try to fit a weighted linear fit to the distances 
                                # and time delays (weighted by correlation), 
                                # with slope = velocity
                                p = np.polyfit(distances, time_dels, 1, 
                                               w=np.asarray(weights))
                            except:
                                # if it doesn't work, try unweighted fit
                                p = np.polyfit(distances, time_dels, 1)
                            
                            # for error estimation, interpolate distance, time delay 
                            # and correlation arrays
                            dist_new = np.linspace(distances[0], distances[-1], 9)
                            time_interp = interp1d(distances, time_dels)
                            weight_interp = interp1d(distances, weights)
                            time_new = time_interp(dist_new)
                            weights_new = weight_interp(dist_new)
                            try:
                                # try to fit interpolated array with covariance to 
                                # get error on slope
                                p,V = np.polyfit(dist_new, time_new, 1, 
                                               w=np.asarray(weights_new), cov=True)
                                pol_err[k][i] = np.sqrt(V[0][0])
                            except:
                                # if it doesn't work, set error as NaN
                                pol_err[k][i] = np.NaN
                            
                            # velocity is 1/slope
                            pol_vels[k][i] = 1.0 / p[0]
                    # append arrays of values for all channels (for one timeslice)
                    mid_time_points.append(mid_time_point)
                    poloidal_velocities.append(pol_vels)
                    poloidal_correlation.append(pol_corr)
                    poloidal_errors.append(pol_err)
            else:
                for j in range(num_slices):
                    # indices for the current slice over which to correlate
                    index1 = idx1 + j * slide_point
                    index2 = index1 + xcorr_length
                    length_chunk = (index2 - index1)
                    
                    # the mid-point of the slice - take this to be the time point for the velocity
                    mid_time_point = 0.5 * (bes_time[index1] + bes_time[index2])
                    
                    # choose the equilibrium value array for the timepoint
                    eq_list_idx = (np.abs(cut_eq_times - mid_time_point)).argmin()
                    eq_array_to_use = eq_array_list[eq_list_idx]
                    
                    # the time base for the time delay estimation
                    time_base = -(1.0 / f_samp) * np.arange(-int(length_chunk / 2), 
                                  int(length_chunk / 2), 1)
                    
                    # the poloidal velocity array for this time slice
                    pol_vels = np.zeros((nrows, ncols))
                    # the associated average correlation values
                    pol_corr = np.zeros((nrows, ncols))
                    # the error on the gradient fit (i.e. the velocity)
                    pol_err = np.zeros((nrows, ncols))
                    
                    # goes through all columns and rows unless otherwise specified
                    for i in cols_to_calc:
                        for k in rows_to_calc:
                            # the reference channel:
                            refch = i + ncols * k
                            # the equilibrium value for the reference channel
                            refch_eq = eq_array_to_use[k, i]
                            # finding four channels to correlate with (including itself)
                            # based on which channel in each row has the closest 
                            # equilibrium value
                            chans = []
                            for row_x in range(nrows):
                                col_x = (np.abs(eq_array_to_use[row_x, :] - 
                                                refch_eq)).argmin()
                                chans.append(col_x + ncols * row_x)
                            
                            # the distances, time delays and cross correlation values 
                            # of the peaks
                            distances = []
                            time_dels = []
                            weights = []
                            
                            # go through the four chosen channels
                            for ch in chans:
                                # the cross-correlation for this channel with the 
                                # reference channel
                                xcorr = cross_corr(refch, ch, bpf_data, index1, 
                                                   index2)
                                # scipy.optimize.curve_fit on freia doesn't like the 
                                # complex data, so convert to float
                                xcorr = xcorr.astype(float)
                                
                                # the distance between channels
                                if ch == refch:
                                    distances.append(0.0)
                                else:
                                    # includes sign for direction
                                    distances.append(
                                        ((ch - refch) / np.abs(ch - refch)) * np.sqrt(
                                            np.abs(apdpos[ch][1] - 
                                                   apdpos[refch][1]) ** 2 + 
                                            np.abs(apdpos[ch][0] - 
                                                   apdpos[refch][0]) ** 2))
                                # the time delay of the peak 
                                # (maximum correlation value)
                                time_del_peak = time_base[np.argmax(xcorr)]
                                # the value of the cross-correlation at peak
                                maxcorr_peak = np.max(xcorr)
                                # choose peak based on maximum correlation
                                time_dels.append(time_del_peak)
                                # assign the correlation values as weights
                                weights.append(maxcorr_peak)
                            
                            # calculate the average correlation
                            av_corr = (np.sum(np.asarray(weights))) / (len(chans))
                            pol_corr[k][i] = av_corr
                            try:
                                # try to fit a weighted linear fit to the distances 
                                # and time delays (weighted by correlation), 
                                # with slope = velocity
                                p = np.polyfit(distances, time_dels, 1, 
                                               w=np.asarray(weights))
                            except:
                                # if it doesn't work, try unweighted fit
                                p = np.polyfit(distances, time_dels, 1)
                            
                            # for error estimation, interpolate distance, time delay 
                            # and correlation arrays
                            dist_new = np.linspace(distances[0], distances[-1], 9)
                            time_interp = interp1d(distances, time_dels)
                            weight_interp = interp1d(distances, weights)
                            time_new = time_interp(dist_new)
                            weights_new = weight_interp(dist_new)
                            try:
                                # try to fit interpolated array with covariance to 
                                # get error on slope
                                p,V = np.polyfit(dist_new, time_new, 1, 
                                               w=np.asarray(weights_new), cov=True)
                                pol_err[k][i] = np.sqrt(V[0][0])
                            except:
                                # if it doesn't work, set error as NaN
                                pol_err[k][i] = np.NaN
                            
                            # velocity is 1/slope
                            pol_vels[k][i] = 1.0 / p[0]
                    # append arrays of values for all channels (for one timeslice)
                    mid_time_points.append(mid_time_point)
                    poloidal_velocities.append(pol_vels)
                    poloidal_correlation.append(pol_corr)
                    poloidal_errors.append(pol_err)
            print("Finished poloidal velocity calculations.")
            results_dict["poloidal"]["mid_time_points"] = mid_time_points
            results_dict["poloidal"]["velocities"] = poloidal_velocities
            results_dict["poloidal"]["correlations"] = poloidal_correlation
            results_dict["poloidal"]["errors"] = poloidal_errors
        if calc_radial is True:
            results_dict["radial"] = {}
            
            eq_rad_chans_list = []
            for eq_idx in range(len(cut_eq_times)):
                # make an array of chosen radial channels for each reference BES 
                # channel based on equilibrium
                eq_timepoint = cut_eq_times[eq_idx]
                equilib = eq.equilibrium(device='MAST', shot=shot, 
                                         time=eq_timepoint)
                eq_rad_chans = calc_radial_chans_from_equilibrium(
                    equilib.R, equilib.Z, np.sqrt(equilib.psiN), apdpos, BESdata)
                eq_rad_chans_list.append(eq_rad_chans)
            print("Radial channels chosen based on equilibrium values.")
            print("Calculating radial velocities...")
            
            mid_time_points = []
            radial_velocities = []
            radial_correlation = []
            radial_errors = []
            
            if use_gaussian_fit is True:
                for j in range(num_slices):
                    # indices for the current slice over which to correlate
                    index1 = idx1 + j * slide_point
                    index2 = index1 + xcorr_length
                    length_chunk = (index2 - index1)
                    
                    # the mid-point of the slice - take this to be the time point for the velocity
                    mid_time_point = 0.5 * (bes_time[index1] + bes_time[index2])
                    
                    # get array of chosen radial channels for each reference BES 
                    # channel based on equilibrium
                    eq_list_idx = (np.abs(cut_eq_times - mid_time_point)).argmin()
                    eq_rad_chans_to_use = eq_rad_chans_list[eq_list_idx]
                    
                    # the time base for the time delay estimation
                    time_base = -(1.0 / f_samp) * np.arange(-int(length_chunk / 2), 
                                  int(length_chunk / 2), 1)
                    
                    # the radial velocity array for this time slice
                    rad_vels = np.zeros((nrows, ncols))
                    # the associated average correlation values
                    rad_corr = np.zeros((nrows, ncols))
                    # the error on the gradient fit (i.e. the velocity)
                    rad_err = np.zeros((nrows, ncols))
                    
                    # goes through all columns and rows unless otherwise specified
                    for k in rows_to_calc:
                        for i in cols_to_calc:
                            # the reference channel:
                            refch = i + ncols * k
                            
                            # the distances, time delays and cross correlation values 
                            # of the peaks
                            distances = []
                            time_dels = []
                            weights = []
                            
                            # this sets the calculation to use only the nearest 
                            # neighbour channels.. do I want to add an option to use 
                            # more channels? or otherwise fix the previous calculation 
                            # of chosen channels to only consider nearest neighbour 
                            # columns..
                            errors = True
                            chans_1 = []
                            for ccc in range(ncols):
                                if eq_rad_chans_to_use[refch, ccc] > -1:
                                    chans_1.append(eq_rad_chans_to_use[refch, ccc])
                            rc_idx = chans_1.index(refch)
                            if rc_idx > 0:
                                if rc_idx < len(chans_1) - 1:
                                    chans = [chans_1[rc_idx - 1], chans_1[rc_idx], chans_1[rc_idx + 1]]
                                else:
                                    if rc_idx > 1:
                                        chans = [chans_1[rc_idx - 2], chans_1[rc_idx - 1], chans_1[rc_idx]]
                                    else:
                                        chans = [chans_1[rc_idx - 1], chans_1[rc_idx]]
                            else:
                                if len(chans_1) > 2:
                                    chans = [chans_1[rc_idx], chans_1[rc_idx + 1], chans_1[rc_idx + 2]]
                                else:
                                    chans = chans_1
                            if len(chans) < 3:
                                errors = False
                            
                            # go through the four chosen channels
                            for ch in chans:
                                # the cross-correlation for this channel with the 
                                # reference channel
                                xcorr = cross_corr(refch, ch, bpf_data, index1, 
                                                   index2)
                                # scipy.optimize.curve_fit on freia doesn't like the 
                                # complex data, so convert to float
                                xcorr = xcorr.astype(float)
                                
                                # the distance between channels
                                if ch == refch:
                                    distances.append(0.0)
                                else:
                                    # includes sign for direction
                                    distances.append(
                                        ((ch - refch) / np.abs(ch - refch)) * np.sqrt(
                                            np.abs(apdpos[ch][1] - 
                                                   apdpos[refch][1]) ** 2 + 
                                            np.abs(apdpos[ch][0] - 
                                                   apdpos[refch][0]) ** 2))
                                # the time delay of the peak 
                                # (maximum correlation value)
                                time_del_peak = time_base[np.argmax(xcorr)]
                                # the value of the cross-correlation at peak
                                maxcorr_peak = np.max(xcorr)
                                
                                # the time base index of the peak
                                peak_idx = np.argwhere(time_base == 
                                                       time_del_peak)[0][0]
                                
                                # find the start and end indices of the slice over 
                                # which to perform a Gaussian fit, based on the 
                                # fit_range parameter
                                idx_stop = peak_idx - int((fit_range / 2) * length_chunk)
                                if idx_stop < 0:
                                    idx_stop = 0
                                
                                idx_start = peak_idx + int((fit_range / 2) * length_chunk)
                                if idx_start > len(time_base) - 1:
                                    idx_start = len(time_base) - 1
                                
                                # need to flip the time base and the correlation values
                                rev_time = np.flip(time_base[idx_stop:idx_start])
                                rev_corr = np.flip(xcorr[idx_stop:idx_start])
                                
                                try:
                                    # try to fit a Gaussian
                                    popt, pcov = curve_fit(
                                        gaussian, rev_time, rev_corr, p0=[
                                            maxcorr_peak, time_del_peak, 
                                            (rev_time[-1] - rev_time[0]) / 2.0])
                                    # choose new peak time and correlation, based on 
                                    # Gaussian fit
                                    time_val = rev_time[np.argmax(gaussian(
                                        rev_time, *popt))]
                                    corr_val = np.max(gaussian(rev_time, *popt))
                                except:
                                    # if fit does not work, choose peak based on 
                                    # maxixmum correlation
                                    time_val = time_del_peak
                                    corr_val = maxcorr_peak
                                time_dels.append(time_val)
                                # assign the correlation values as weights
                                weights.append(corr_val)
                            
                            # calculate the average correlation
                            av_corr = (np.sum(np.asarray(weights))) / (len(chans))
                            rad_corr[k][i] = av_corr
                            try:
                                # try to fit a weighted linear fit to the distances 
                                # and time delays (weighted by correlation)
                                p = np.polyfit(distances, time_dels, 1, 
                                               w=np.asarray(weights))
                                # velocity is 1/slope
                                rad_vels[k][i] = 1.0 / p[0]
                            except:
                                try:
                                    # if it doesn't work, try unweighted fit
                                    p = np.polyfit(distances, time_dels, 1)
                                    rad_vels[k][i] = 1.0 / p[0]
                                except:
                                    # if no fit is possible (e.g. not enough channels)
                                    rad_vels[k][i] = np.NaN
                            
                            if errors:
                                # for error estimation, interpolate distance, time delay 
                                # and correlation arrays
                                dist_new = np.linspace(distances[0], distances[-1], 9)
                                time_interp = interp1d(distances, time_dels)
                                weight_interp = interp1d(distances, weights)
                                time_new = time_interp(dist_new)
                                weights_new = weight_interp(dist_new)
                                try:
                                    # try to fit interpolated array with covariance to 
                                    # get error on slope
                                    p,V = np.polyfit(dist_new, time_new, 1, 
                                                   w=np.asarray(weights_new), cov=True)
                                    rad_err[k][i] = np.sqrt(V[0][0])
                                except:
                                    # if it doesn't work, set error as NaN
                                    rad_err[k][i] = np.NaN
                            else:
                                rad_err[k][i] = np.NaN
                            
                    # append arrays of values for all channels (for one timeslice)
                    mid_time_points.append(mid_time_point)
                    radial_velocities.append(rad_vels)
                    radial_correlation.append(rad_corr)
                    radial_errors.append(rad_err)
            else:
                for j in range(num_slices):
                    # indices for the current slice over which to correlate
                    index1 = idx1 + j * slide_point
                    index2 = index1 + xcorr_length
                    length_chunk = (index2 - index1)
                    
                    # the mid-point of the slice - take this to be the time point for the velocity
                    mid_time_point = 0.5 * (bes_time[index1] + bes_time[index2])
                    
                    # get array of chosen radial channels for each reference BES 
                    # channel based on equilibrium
                    eq_list_idx = (np.abs(cut_eq_times - mid_time_point)).argmin()
                    eq_rad_chans_to_use = eq_rad_chans_list[eq_list_idx]
                    
                    # the time base for the time delay estimation
                    time_base = -(1.0 / f_samp) * np.arange(-int(length_chunk / 2), 
                                  int(length_chunk / 2), 1)
                    
                    # the radial velocity array for this time slice
                    rad_vels = np.zeros((nrows, ncols))
                    # the associated average correlation values
                    rad_corr = np.zeros((nrows, ncols))
                    # the error on the gradient fit (i.e. the velocity)
                    rad_err = np.zeros((nrows, ncols))
                    
                    # goes through all columns and rows unless otherwise specified
                    for k in rows_to_calc:
                        for i in cols_to_calc:
                            # the reference channel:
                            refch = i + ncols * k
                            
                            # the distances, time delays and cross correlation values 
                            # of the peaks
                            distances = []
                            time_dels = []
                            weights = []
                            
                            # this sets the calculation to use only the nearest 
                            # neighbour channels.. do I want to add an option to use 
                            # more channels? or otherwise fix the previous calculation 
                            # of chosen channels to only consider nearest neighbour 
                            # columns..
                            errors = True
                            chans_1 = []
                            for ccc in range(ncols):
                                if eq_rad_chans_to_use[refch, ccc] > -1:
                                    chans_1.append(eq_rad_chans_to_use[refch, ccc])
                            rc_idx = chans_1.index(refch)
                            if rc_idx > 0:
                                if rc_idx < len(chans_1) - 1:
                                    chans = [chans_1[rc_idx - 1], chans_1[rc_idx], chans_1[rc_idx + 1]]
                                else:
                                    if rc_idx > 1:
                                        chans = [chans_1[rc_idx - 2], chans_1[rc_idx - 1], chans_1[rc_idx]]
                                    else:
                                        chans = [chans_1[rc_idx - 1], chans_1[rc_idx]]
                            else:
                                if len(chans_1) > 2:
                                    chans = [chans_1[rc_idx], chans_1[rc_idx + 1], chans_1[rc_idx + 2]]
                                else:
                                    chans = chans_1
                            if len(chans) < 3:
                                errors = False
                            
                            # go through the four chosen channels
                            for ch in chans:
                                # the cross-correlation for this channel with the 
                                # reference channel
                                xcorr = cross_corr(refch, ch, bpf_data, index1, 
                                                   index2)
                                # scipy.optimize.curve_fit on freia doesn't like the 
                                # complex data, so convert to float
                                xcorr = xcorr.astype(float)
                                
                                # the distance between channels
                                if ch == refch:
                                    distances.append(0.0)
                                else:
                                    # includes sign for direction
                                    distances.append(
                                        ((ch - refch) / np.abs(ch - refch)) * np.sqrt(
                                            np.abs(apdpos[ch][1] - 
                                                   apdpos[refch][1]) ** 2 + 
                                            np.abs(apdpos[ch][0] - 
                                                   apdpos[refch][0]) ** 2))
                                # the time delay of the peak 
                                # (maximum correlation value)
                                time_del_peak = time_base[np.argmax(xcorr)]
                                # the value of the cross-correlation at peak
                                maxcorr_peak = np.max(xcorr)
                                # choose peak based on maxixmum correlation
                                time_dels.append(time_del_peak)
                                # assign the correlation values as weights
                                weights.append(maxcorr_peak)
                            
                            # calculate the average correlation
                            av_corr = (np.sum(np.asarray(weights))) / (len(chans))
                            rad_corr[k][i] = av_corr
                            try:
                                # try to fit a weighted linear fit to the distances 
                                # and time delays (weighted by correlation)
                                p = np.polyfit(distances, time_dels, 1, 
                                               w=np.asarray(weights))
                                # velocity is 1/slope
                                rad_vels[k][i] = 1.0 / p[0]
                            except:
                                try:
                                    # if it doesn't work, try unweighted fit
                                    p = np.polyfit(distances, time_dels, 1)
                                    rad_vels[k][i] = 1.0 / p[0]
                                except:
                                    # if no fit is possible (e.g. not enough channels)
                                    rad_vels[k][i] = np.NaN
                            
                            if errors:
                                # for error estimation, interpolate distance, time delay 
                                # and correlation arrays
                                dist_new = np.linspace(distances[0], distances[-1], 9)
                                time_interp = interp1d(distances, time_dels)
                                weight_interp = interp1d(distances, weights)
                                time_new = time_interp(dist_new)
                                weights_new = weight_interp(dist_new)
                                try:
                                    # try to fit interpolated array with covariance to 
                                    # get error on slope
                                    p,V = np.polyfit(dist_new, time_new, 1, 
                                                   w=np.asarray(weights_new), cov=True)
                                    rad_err[k][i] = np.sqrt(V[0][0])
                                except:
                                    # if it doesn't work, set error as NaN
                                    rad_err[k][i] = np.NaN
                            else:
                                rad_err[k][i] = np.NaN
                            
                    # append arrays of values for all channels (for one timeslice)
                    mid_time_points.append(mid_time_point)
                    radial_velocities.append(rad_vels)
                    radial_correlation.append(rad_corr)
                    radial_errors.append(rad_err)
            print("Finished radial velocity calculations.")
            results_dict["radial"]["mid_time_points"] = mid_time_points
            results_dict["radial"]["velocities"] = radial_velocities
            results_dict["radial"]["correlations"] = radial_correlation
            results_dict["radial"]["errors"] = radial_errors
            
    return results_dict

# made a separate function for the case of removing poloidal average and 
# calculating both radial and poloidal velocities, as it's slightly faster to 
# calculate them both in the same loop
def calc_radial_and_poloidal_with_removed_average(
        bpf_data, cut_eq_times, apdpos, use_gaussian_fit, num_slices, 
        slide_point, xcorr_length, idx1, bes_time, eq_array_list, 
        eq_rad_chans_list, ncols, nrows, 
        f_samp, cols_to_calc, rows_to_calc, fit_range):
    results_dict = {"poloidal": {}, "radial": {}}
    
    new_fluct_data = np.copy(bpf_data)
    
    mid_time_points = []
    poloidal_velocities = []
    poloidal_correlation = []
    poloidal_errors = []
    radial_velocities = []
    radial_correlation = []
    radial_errors = []
    
    if use_gaussian_fit is True:
        for j in range(num_slices):
            # indices for the current slice over which to correlate
            index1 = idx1 + j * slide_point
            index2 = index1 + xcorr_length
            length_chunk = (index2 - index1)
            
            # the mid-point of the slice - take this to be the time point for the velocity
            mid_time_point = 0.5 * (bes_time[index1] + bes_time[index2])
            
            # choose the equilibrium value array for the timepoint
            eq_list_idx = (np.abs(cut_eq_times - mid_time_point)).argmin()
            eq_array_to_use = eq_array_list[eq_list_idx]
            poloidal_channels = {}
            for i_x in range(ncols):
                for k_x in range(nrows):
                    refch_x = i_x + ncols * k_x
                    # find the poloidal channels for each channel
                    refch_eq = eq_array_to_use[k_x, i_x]
                    chans_pol = []
                    for row_x in range(nrows):
                        col_x = (np.abs(eq_array_to_use[row_x, :] - 
                                        refch_eq)).argmin()
                        chans_pol.append(col_x + ncols * row_x)
                    poloidal_channels[str(refch_x)] = chans_pol
                    
                    # find the poloidal average
                    average = (bpf_data[chans_pol[0], :] + 
                               bpf_data[chans_pol[1], :] + 
                               bpf_data[chans_pol[2], :] + 
                               bpf_data[chans_pol[3], :]) / 4.0
                    # subtract poloidal average from the data
                    new_fluct_data[refch_x, :] = bpf_data[refch_x, :] - average
            
            # get array of chosen radial channels for each reference BES 
            # channel based on equilibrium
            eq_rad_chans_to_use = eq_rad_chans_list[eq_list_idx]
            
            # the time base for the time delay estimation
            time_base = -(1.0 / f_samp) * np.arange(-int(length_chunk / 2), 
                          int(length_chunk / 2), 1)
            
            # the poloidal and radial velocity arrays for this time slice
            pol_vels = np.zeros((nrows, ncols))
            rad_vels = np.zeros((nrows, ncols))
            # the associated average correlation values
            pol_corr = np.zeros((nrows, ncols))
            rad_corr = np.zeros((nrows, ncols))
            # the error on the gradient fit (i.e. on the velocity)
            pol_err = np.zeros((nrows, ncols))
            rad_err = np.zeros((nrows, ncols))
            
            # goes through all columns and rows unless otherwise specified
            for i in cols_to_calc:
                for k in rows_to_calc:
                    # the reference channel:
                    refch = i + ncols * k
                    
                    ###
                    # calculating the poloidal velocity...
                    # the equilibrium value for the reference channel
                    refch_eq = eq_array_to_use[k, i]
                    # finding four channels to correlate with (including itself)
                    # based on which channel in each row has the closest 
                    # equilibrium value
                    chans_poloidal = poloidal_channels[str(refch)]
                    # the distances, time delays and cross correlation values of the peaks
                    distances_poloidal = []
                    time_dels_poloidal = []
                    weights_poloidal = []
                    # go through the chosen channels
                    for ch in chans_poloidal:
                        # the cross-correlation for this channel with the reference channel
                        xcorr = cross_corr(refch, ch, new_fluct_data, index1, index2)
                        # scipy.optimize.curve_fit on freia doesn't like the complex data, 
                        # so convert to float
                        xcorr = xcorr.astype(float)
                        
                        # the distance between channels
                        if ch == refch:
                            distances_poloidal.append(0.0)
                        else:
                            # includes sign for direction
                            distances_poloidal.append(
                                ((ch - refch) / np.abs(ch - refch)) * 
                                np.sqrt(np.abs(apdpos[ch][1] - apdpos[refch][1]) ** 2 + 
                                        np.abs(apdpos[ch][0] - apdpos[refch][0]) ** 2))
                        # the time delay of the peak (maximum correlation value)
                        time_del_peak = time_base[np.argmax(xcorr)]
                        # the value of the cross-correlation at peak
                        maxcorr_peak = np.max(xcorr)
                        
                        # the time base index of the peak
                        peak_idx = np.argwhere(time_base == time_del_peak)[0][0]
                        
                        # find the start and end indices of the slice over which to perform a 
                        # Gaussian fit, based on the fit_range parameter
                        idx_stop = peak_idx - int((fit_range / 2) * length_chunk)
                        if idx_stop < 0:
                            idx_stop = 0
                        
                        idx_start = peak_idx + int((fit_range / 2) * length_chunk)
                        if idx_start > len(time_base) - 1:
                            idx_start = len(time_base) - 1
                        
                        # need to flip the time base and the correlation values
                        rev_time = np.flip(time_base[idx_stop:idx_start])
                        rev_corr = np.flip(xcorr[idx_stop:idx_start])
                        
                        try:
                            # try to fit a Gaussian
                            popt, pcov = curve_fit(
                                gaussian, rev_time, rev_corr, p0=[
                                    maxcorr_peak, time_del_peak, 
                                    (rev_time[-1] - rev_time[0]) / 2.0])
                            # choose new peak time and correlation, based on Gaussian fit
                            time_val = rev_time[np.argmax(gaussian(rev_time, *popt))]
                            corr_val = np.max(gaussian(rev_time, *popt))
                        except:
                            # if fit does not work, choose peak based on maxixmum correlation
                            time_val = time_del_peak
                            corr_val = maxcorr_peak
                        time_dels_poloidal.append(time_val)
                        # assign the correlation values as weights
                        weights_poloidal.append(corr_val)
                    
                    # calculate the average correlation
                    pol_corr[k][i] = (np.sum(
                        np.asarray(weights_poloidal))) / 4.0
                    try:
                        # try to fit a weighted linear fit to the distances 
                        # and time delays (weighted by correlation), 
                        # with slope = velocity
                        p_pol = np.polyfit(distances_poloidal, time_dels_poloidal, 1, 
                                       w=np.asarray(weights_poloidal))
                    except:
                        # if it doesn't work, try unweighted fit
                        p_pol = np.polyfit(distances_poloidal, time_dels_poloidal, 1)
                    # velocity is 1/slope
                    pol_vels[k][i] = 1.0 / p_pol[0]
                    
                    # for error estimation, interpolate distance, time delay 
                    # and correlation arrays
                    dist_new_pol = np.linspace(distances_poloidal[0], distances_poloidal[-1], 9)
                    time_pol_interp = interp1d(distances_poloidal, time_dels_poloidal)
                    weight_pol_interp = interp1d(distances_poloidal, weights_poloidal)
                    time_new_pol = time_pol_interp(dist_new_pol)
                    weights_new_pol = weight_pol_interp(dist_new_pol)
                    try:
                        # try to fit interpolated array with covariance to 
                        # get error on slope
                        p,V = np.polyfit(dist_new_pol, time_new_pol, 1, 
                                       w=np.asarray(weights_new_pol), cov=True)
                        pol_err[k][i] = np.sqrt(V[0][0])
                    except:
                        # if it doesn't work, set error as NaN
                        pol_err[k][i] = np.NaN
                    
                    ###
                    # calculating the radial velocity...
                    
                    # this sets the calculation to use only the nearest 
                    # neighbour channels.. do I want to add an option to use 
                    # more channels? or otherwise fix the previous calculation 
                    # of chosen channels to only consider nearest neighbour 
                    # columns..
                    errors = True
                    chans_1 = []
                    for ccc in range(ncols):
                        if eq_rad_chans_to_use[refch, ccc] > -1:
                            chans_1.append(eq_rad_chans_to_use[refch, ccc])
                    rc_idx = chans_1.index(refch)
                    if rc_idx > 0:
                        if rc_idx < len(chans_1) - 1:
                            chans_radial = [chans_1[rc_idx - 1], chans_1[rc_idx], chans_1[rc_idx + 1]]
                        else:
                            if rc_idx > 1:
                                chans_radial = [chans_1[rc_idx - 2], chans_1[rc_idx - 1], chans_1[rc_idx]]
                            else:
                                chans_radial = [chans_1[rc_idx - 1], chans_1[rc_idx]]
                    else:
                        if len(chans_1) > 2:
                            chans_radial = [chans_1[rc_idx], chans_1[rc_idx + 1], chans_1[rc_idx + 2]]
                        else:
                            chans_radial = chans_1
                    if len(chans_radial) < 3:
                        errors = False
                    
                    # the distances, time delays and cross correlation values of the peaks
                    distances_radial = []
                    time_dels_radial = []
                    weights_radial = []
                    # go through the chosen channels
                    for ch in chans_radial:
                        # the cross-correlation for this channel with the reference channel
                        xcorr = cross_corr(refch, ch, new_fluct_data, index1, index2)
                        # scipy.optimize.curve_fit on freia doesn't like the complex data, 
                        # so convert to float
                        xcorr = xcorr.astype(float)
                        
                        # the distance between channels
                        if ch == refch:
                            distances_radial.append(0.0)
                        else:
                            # includes sign for direction
                            distances_radial.append(
                                ((ch - refch) / np.abs(ch - refch)) * 
                                np.sqrt(np.abs(apdpos[ch][1] - apdpos[refch][1]) ** 2 + 
                                        np.abs(apdpos[ch][0] - apdpos[refch][0]) ** 2))
                        # the time delay of the peak (maximum correlation value)
                        time_del_peak = time_base[np.argmax(xcorr)]
                        # the value of the cross-correlation at peak
                        maxcorr_peak = np.max(xcorr)
                        
                        # the time base index of the peak
                        peak_idx = np.argwhere(time_base == time_del_peak)[0][0]
                        
                        # find the start and end indices of the slice over which to perform a 
                        # Gaussian fit, based on the fit_range parameter
                        idx_stop = peak_idx - int((fit_range / 2) * length_chunk)
                        if idx_stop < 0:
                            idx_stop = 0
                        
                        idx_start = peak_idx + int((fit_range / 2) * length_chunk)
                        if idx_start > len(time_base) - 1:
                            idx_start = len(time_base) - 1
                        
                        # need to flip the time base and the correlation values
                        rev_time = np.flip(time_base[idx_stop:idx_start])
                        rev_corr = np.flip(xcorr[idx_stop:idx_start])
                        
                        try:
                            # try to fit a Gaussian
                            popt, pcov = curve_fit(
                                gaussian, rev_time, rev_corr, p0=[
                                    maxcorr_peak, time_del_peak, 
                                    (rev_time[-1] - rev_time[0]) / 2.0])
                            # choose new peak time and correlation, based on Gaussian fit
                            time_val = rev_time[np.argmax(gaussian(rev_time, *popt))]
                            corr_val = np.max(gaussian(rev_time, *popt))
                        except:
                            # if fit does not work, choose peak based on maxixmum correlation
                            time_val = time_del_peak
                            corr_val = maxcorr_peak
                        time_dels_radial.append(time_val)
                        # assign the correlation values as weights
                        weights_radial.append(corr_val)
                    
                    # calculate the average correlation
                    rad_corr[k][i] = (
                        np.sum(np.asarray(weights_radial))) / len(chans_radial)
                    
                    try:
                        # try to fit a weighted linear fit to the distances 
                        # and time delays (weighted by correlation)
                        p_rad = np.polyfit(distances_radial, time_dels_radial, 1, 
                                       w=np.asarray(weights_radial))
                        # velocity is 1/slope
                        rad_vels[k][i] = 1.0 / p_rad[0]
                    except:
                        try:
                            # if it doesn't work, try unweighted fit
                            p_rad = np.polyfit(distances_radial, time_dels_radial, 1)
                            rad_vels[k][i] = 1.0 / p_rad[0]
                        except:
                            # if no fit is possible (e.g. not enough channels)
                            rad_vels[k][i] = np.NaN
                    
                    if errors:
                        # for error estimation, interpolate distance, time delay 
                        # and correlation arrays
                        dist_new_rad = np.linspace(distances_radial[0], distances_radial[-1], 9)
                        time_rad_interp = interp1d(distances_radial, time_dels_radial)
                        weight_rad_interp = interp1d(distances_radial, weights_radial)
                        time_new_rad = time_rad_interp(dist_new_rad)
                        weights_new_rad = weight_rad_interp(dist_new_rad)
                        try:
                            # try to fit interpolated array with covariance to 
                            # get error on slope
                            p,V = np.polyfit(dist_new_rad, time_new_rad, 1, 
                                           w=np.asarray(weights_new_rad), cov=True)
                            rad_err[k][i] = np.sqrt(V[0][0])
                        except:
                            # if it doesn't work, set error as NaN
                            rad_err[k][i] = np.NaN
                    else:
                        rad_err[k][i] = np.NaN
            # append arrays of values for all channels (for one timeslice)
            mid_time_points.append(mid_time_point)
            poloidal_velocities.append(pol_vels)
            poloidal_correlation.append(pol_corr)
            poloidal_errors.append(pol_err)
            radial_velocities.append(rad_vels)
            radial_correlation.append(rad_corr)
            radial_errors.append(rad_err)
    else:
        for j in range(num_slices):
            # indices for the current slice over which to correlate
            index1 = idx1 + j * slide_point
            index2 = index1 + xcorr_length
            length_chunk = (index2 - index1)
            
            # the mid-point of the slice - take this to be the time point for the velocity
            mid_time_point = 0.5 * (bes_time[index1] + bes_time[index2])
            
            # choose the equilibrium value array for the timepoint
            eq_list_idx = (np.abs(cut_eq_times - mid_time_point)).argmin()
            eq_array_to_use = eq_array_list[eq_list_idx]
            poloidal_channels = {}
            for i_x in range(ncols):
                for k_x in range(nrows):
                    refch_x = i_x + ncols * k_x
                    # find the poloidal channels for each channel
                    refch_eq = eq_array_to_use[k_x, i_x]
                    chans_pol = []
                    for row_x in range(nrows):
                        col_x = (np.abs(eq_array_to_use[row_x, :] - 
                                        refch_eq)).argmin()
                        chans_pol.append(col_x + ncols * row_x)
                    poloidal_channels[str(refch_x)] = chans_pol
                    
                    # find the poloidal average
                    average = (bpf_data[chans_pol[0], :] + 
                               bpf_data[chans_pol[1], :] + 
                               bpf_data[chans_pol[2], :] + 
                               bpf_data[chans_pol[3], :]) / 4.0
                    # subtract poloidal average from the data
                    new_fluct_data[refch_x, :] = bpf_data[refch_x, :] - average
            
            # get array of chosen radial channels for each reference BES 
            # channel based on equilibrium
            eq_rad_chans_to_use = eq_rad_chans_list[eq_list_idx]
            
            # the time base for the time delay estimation
            time_base = -(1.0 / f_samp) * np.arange(-int(length_chunk / 2), 
                          int(length_chunk / 2), 1)
            
            # the poloidal and radial velocity arrays for this time slice
            pol_vels = np.zeros((nrows, ncols))
            rad_vels = np.zeros((nrows, ncols))
            # the associated average correlation values
            pol_corr = np.zeros((nrows, ncols))
            rad_corr = np.zeros((nrows, ncols))
            # the error on the gradient fit (i.e. on the velocity)
            pol_err = np.zeros((nrows, ncols))
            rad_err = np.zeros((nrows, ncols))
            
            # goes through all columns and rows unless otherwise specified
            for i in cols_to_calc:
                for k in rows_to_calc:
                    # the reference channel:
                    refch = i + ncols * k
                    
                    ###
                    # calculating the poloidal velocity...
                    
                    # the equilibrium value for the reference channel
                    refch_eq = eq_array_to_use[k, i]
                    # finding four channels to correlate with (including itself)
                    # based on which channel in each row has the closest 
                    # equilibrium value
                    chans_poloidal = poloidal_channels[str(refch)]
                    
                    # the distances, time delays and cross correlation values 
                    # of the peaks
                    distances_poloidal = []
                    time_dels_poloidal = []
                    weights_poloidal = []
                    
                    # go through the four chosen channels
                    for ch in chans_poloidal:
                        # the cross-correlation for this channel with the 
                        # reference channel
                        xcorr = cross_corr(refch, ch, new_fluct_data, index1, 
                                           index2)
                        # scipy.optimize.curve_fit on freia doesn't like the 
                        # complex data, so convert to float
                        xcorr = xcorr.astype(float)
                        
                        # the distance between channels
                        if ch == refch:
                            distances_poloidal.append(0.0)
                        else:
                            # includes sign for direction
                            distances_poloidal.append(
                                ((ch - refch) / np.abs(ch - refch)) * 
                                np.sqrt(
                                    np.abs(apdpos[ch][1] - 
                                           apdpos[refch][1]) ** 2 + 
                                    np.abs(apdpos[ch][0] - 
                                           apdpos[refch][0]) ** 2))
                        # the time delay of the peak 
                        # (maximum correlation value)
                        time_del_peak = time_base[np.argmax(xcorr)]
                        # the value of the cross-correlation at peak
                        maxcorr_peak = np.max(xcorr)
                        # choose peak based on maxixmum correlation
                        time_dels_poloidal.append(time_del_peak)
                        # assign the correlation values as weights
                        weights_poloidal.append(maxcorr_peak)
                    
                    # calculate the average correlation
                    pol_corr[k][i] = (np.sum(
                        np.asarray(weights_poloidal))) / 4.0
                    try:
                        # try to fit a weighted linear fit to the distances 
                        # and time delays (weighted by correlation), 
                        # with slope = velocity
                        p_pol = np.polyfit(distances_poloidal, time_dels_poloidal, 1, 
                                       w=np.asarray(weights_poloidal))
                    except:
                        # if it doesn't work, try unweighted fit
                        p_pol = np.polyfit(distances_poloidal, time_dels_poloidal, 1)
                    
                    # velocity is 1/slope
                    pol_vels[k][i] = 1.0 / p_pol[0]
                    
                    # for error estimation, interpolate distance, time delay 
                    # and correlation arrays
                    dist_new_pol = np.linspace(distances_poloidal[0], distances_poloidal[-1], 9)
                    time_pol_interp = interp1d(distances_poloidal, time_dels_poloidal)
                    weight_pol_interp = interp1d(distances_poloidal, weights_poloidal)
                    time_new_pol = time_pol_interp(dist_new_pol)
                    weights_new_pol = weight_pol_interp(dist_new_pol)
                    try:
                        # try to fit interpolated array with covariance to 
                        # get error on slope
                        p_pol,V_pol = np.polyfit(dist_new_pol, time_new_pol, 1, 
                                       w=np.asarray(weights_new_pol), cov=True)
                        pol_err[k][i] = np.sqrt(V_pol[0][0])
                    except:
                        # if it doesn't work, set error as NaN
                        pol_err[k][i] = np.NaN
                        
                    
                    ###
                    # calculating the radial velocity...
                    
                    # this sets the calculation to use only the nearest 
                    # neighbour channels.. do I want to add an option to use 
                    # more channels? or otherwise fix the previous calculation 
                    # of chosen channels to only consider nearest neighbour 
                    # columns..
                    errors = True
                    chans_1 = []
                    for ccc in range(ncols):
                        if eq_rad_chans_to_use[refch, ccc] > -1:
                            chans_1.append(eq_rad_chans_to_use[refch, ccc])
                    rc_idx = chans_1.index(refch)
                    if rc_idx > 0:
                        if rc_idx < len(chans_1) - 1:
                            chans_radial = [chans_1[rc_idx - 1], chans_1[rc_idx], chans_1[rc_idx + 1]]
                        else:
                            if rc_idx > 1:
                                chans_radial = [chans_1[rc_idx - 2], chans_1[rc_idx - 1], chans_1[rc_idx]]
                            else:
                                chans_radial = [chans_1[rc_idx - 1], chans_1[rc_idx]]
                    else:
                        if len(chans_1) > 2:
                            chans_radial = [chans_1[rc_idx], chans_1[rc_idx + 1], chans_1[rc_idx + 2]]
                        else:
                            chans_radial = chans_1
                    if len(chans_radial) < 3:
                        errors = False
                    
                    # the distances, time delays and cross correlation values 
                    # of the peaks
                    distances_radial = []
                    time_dels_radial = []
                    weights_radial = []
                    
                    # go through the chosen channels
                    for ch in chans_radial:
                        # the cross-correlation for this channel with the 
                        # reference channel
                        xcorr = cross_corr(refch, ch, new_fluct_data, index1, 
                                           index2)
                        # scipy.optimize.curve_fit on freia doesn't like the 
                        # complex data, so convert to float
                        xcorr = xcorr.astype(float)
                        
                        # the distance between channels
                        if ch == refch:
                            distances_radial.append(0.0)
                        else:
                            # includes sign for direction
                            distances_radial.append(
                                ((ch - refch) / np.abs(ch - refch)) * 
                                np.sqrt(
                                    np.abs(apdpos[ch][1] - 
                                           apdpos[refch][1]) ** 2 + 
                                    np.abs(apdpos[ch][0] - 
                                           apdpos[refch][0]) ** 2))
                        # the time delay of the peak 
                        # (maximum correlation value)
                        time_del_peak = time_base[np.argmax(xcorr)]
                        # the value of the cross-correlation at peak
                        maxcorr_peak = np.max(xcorr)
                        # choose peak based on maxixmum correlation
                        time_dels_radial.append(time_del_peak)
                        # assign the correlation values as weights
                        weights_radial.append(maxcorr_peak)
                    
                    # calculate the average correlation
                    rad_corr[k][i] = (
                        np.sum(np.asarray(weights_radial))) / len(chans_radial)
                    
                    try:
                        # try to fit a weighted linear fit to the distances 
                        # and time delays (weighted by correlation)
                        p_rad = np.polyfit(distances_radial, time_dels_radial, 1, 
                                       w=np.asarray(weights_radial))
                        # velocity is 1/slope
                        rad_vels[k][i] = 1.0 / p_rad[0]
                    except:
                        try:
                            # if it doesn't work, try unweighted fit
                            p_rad = np.polyfit(distances_radial, time_dels_radial, 1)
                            rad_vels[k][i] = 1.0 / p_rad[0]
                        except:
                            # if no fit is possible (e.g. not enough channels)
                            rad_vels[k][i] = np.NaN
                    
                    if errors:
                        # for error estimation, interpolate distance, time delay 
                        # and correlation arrays
                        dist_new_rad = np.linspace(distances_radial[0], distances_radial[-1], 9)
                        time_rad_interp = interp1d(distances_radial, time_dels_radial)
                        weight_rad_interp = interp1d(distances_radial, weights_radial)
                        time_new_rad = time_rad_interp(dist_new_rad)
                        weights_new_rad = weight_rad_interp(dist_new_rad)
                        try:
                            # try to fit interpolated array with covariance to 
                            # get error on slope
                            p_rad,V_rad = np.polyfit(dist_new_rad, time_new_rad, 1, 
                                           w=np.asarray(weights_new_rad), cov=True)
                            rad_err[k][i] = np.sqrt(V_rad[0][0])
                        except:
                            # if it doesn't work, set error as NaN
                            rad_err[k][i] = np.NaN
                    else:
                        rad_err[k][i] = np.NaN
            # append arrays of values for all channels (for one timeslice)
            mid_time_points.append(mid_time_point)
            poloidal_velocities.append(pol_vels)
            poloidal_correlation.append(pol_corr)
            poloidal_errors.append(pol_err)
            radial_velocities.append(rad_vels)
            radial_correlation.append(rad_corr)
            radial_errors.append(rad_err)
    results_dict["poloidal"]["mid_time_points"] = mid_time_points
    results_dict["poloidal"]["velocities"] = poloidal_velocities
    results_dict["poloidal"]["correlations"] = poloidal_correlation
    results_dict["poloidal"]["errors"] = poloidal_errors
    results_dict["radial"]["mid_time_points"] = mid_time_points
    results_dict["radial"]["velocities"] = radial_velocities
    results_dict["radial"]["correlations"] = radial_correlation
    results_dict["radial"]["errors"] = radial_errors
    return results_dict
