# copied one for one from /home/lhowlett/usefulcodes/BES_analysis/spike_remover.py

import numpy as np
import matplotlib.pyplot as plt


def spike_filter(signal, time=None, limit=0.5, plot=0):
    """ Remove points in the data series that have gradients (up and down) larger than a specified limit. Copy of Y-c's bes_filter_radiation.pro from IDL.
        Inputs:
            signal      -   [nt] the time series to analyse.
            time        -   [nt] array of timesteps for plotting (not used for filtering so can be dummy)
            limit       -   the numerical size of the gradient above which the 'spike' is removed.
        Outputs:
            signal      -   [nt] same as - signal - but with spikes removed and replaces by the nearest neighbour values.
        """

    nt = len(signal)

    diff = signal[1:nt] - signal[0:(nt-1)]

    n_pulses = 0

    if plot == 1:
        plt.figure()
        plt.plot(time, signal, label="raw")

    # Drop points when diff is one up, one down
    ind = np.where((np.roll(diff,-1) > limit) & (np.roll(diff,-2) < -limit))
    if (ind[0] >= 0).all():
        indind = np.where(ind[0] < len(diff)-6)
        if (indind[0] >= 0).all():
            ind = ind[0]
            ind = ind[indind[0]]
            signal[ind+1] = signal[ind]
            signal[ind+2] = signal[ind]
            signal[ind+3] = signal[ind+4]
            n_pulses = n_pulses+len(ind)

    if plot == 1:
        plt.plot(time, signal, label="1u1d")


    # Drop points with two up and one down
    ind = np.where((np.roll(diff,-1) > limit) & (np.roll(diff,-2) > limit) & (np.roll(diff,-3) < -limit) )
    if (ind[0] >= 0).all():
        indind = np.where(ind[0] < len(diff)-7)
        if (indind[0] >= 0).all():
            ind = ind[0]
            ind = ind[indind[0]]
            signal[ind+1] = signal[ind]
            signal[ind+2] = signal[ind]
            signal[ind+3] = signal[ind+5]
            signal[ind+4] = signal[ind+5]
            n_pulses = n_pulses+len(ind)

    if plot == 1:
        plt.plot(time, signal, label="2u1d")


    # Drop points with one up and two down
    ind = np.where((np.roll(diff,-1) > limit) & (np.roll(diff,-2) < -limit) & (np.roll(diff,-3) < -limit) )
    if (ind[0] >= 0).all():
        indind = np.where(ind[0] < len(diff)-7)
        if (indind[0] >= 0).all():
            ind = ind[0]
            ind = ind[indind[0]]
            signal[ind+1] = signal[ind]
            signal[ind+2] = signal[ind]
            signal[ind+3] = signal[ind+5]
            signal[ind+4] = signal[ind+5]
            n_pulses = n_pulses+len(ind)

    if plot == 1:
        plt.plot(time, signal, label="1u2d")


    # Drop points with one up, one within limit, one one down
    ind = np.where((np.roll(diff,-1) > limit) & (np.roll(abs(diff),-2) < limit) & (np.roll(diff,-3) < -limit))
    if (ind[0] >= 0).all():
        indind = np.where(ind[0] < len(diff)-7)
        if (indind[0] >= 0).all():
            ind = ind[0]
            ind = ind[indind[0]]
            signal[ind+1] = signal[ind]
            signal[ind+2] = signal[ind]
            signal[ind+3] = signal[ind+5]
            signal[ind+4] = signal[ind+5]
            n_pulses = n_pulses+len(ind)

    if plot == 1:
        plt.plot(time, signal, label="1u1in1d")
        plt.legend(loc="upper right")
        plt.show()
        
    return signal, n_pulses

