# File containing the BES data class BESClass along with functions required to 
# obtain the data and calculate the APD array positions and the fluctuation data
from imports import *

class Sig():
    def _init_(self):
        pass

# create a class for BES data
class BESClass:
    def __init__(self, shot_number, r_pos, z_pos, time_data, data, 
                 ss_nbi_time, ss_nbi_data, ncols=8, nrows=4):
        self.shot_number = shot_number      # the shot number
        self.view_radius = r_pos            # the R position of the central view
        self.z_offset = z_pos               # the z offset from the midplane
        self.time = Sig()                   # collect different time arrays
        self.data = Sig()                   # collect different BES data arrays
        self.time.raw = time_data           # the time array for the uncut BES data
        self.data.raw = data                # the uncut data array
        self.nchan = len(data[:,0])         # the number of channels in the array
        self.ncols = ncols                  # number of columns in the BES array
        self.nrows = nrows                  # number of rows in the BES array
        # the sampling frequency of the BES data
        self.f_samp = np.shape(time_data)[0]/(time_data[-1]-time_data[0]) 
        
        self.nbi_times = Sig()              # collect beam start and end times
        
        # for given number of rows and columns of the array, get a list of the
        # channels for each row and column
        columns = []
        for i in range(self.ncols):
            col = []
            for j in range(self.nrows):
                ch = i + j*self.ncols
                col.append(ch)
            columns.append(col)
        
        rows = []
        for i in range(self.nrows):
            row = []
            for j in range(self.ncols):
                ch = self.ncols*i + j
                row.append(ch)
            rows.append(row)
        self.column_list = columns      # list of channels for each column
        self.row_list = rows            # list of channels for each row
        
        # calculate R,Z coordinates of each pixel of the detector
        rsep = 2.3e-3  # active area separation on array
        aa = 1.6e-3  # active area size (length) [mm]
        nommag = 0.02 / 0.0023  # scaling of array:nominal view size in plasma
        
        # position the active area grid, including scaling the nominal vectors
        r_cols = np.multiply(np.arange((-self.ncols/2*rsep) + rsep/2,
                                       (self.ncols/2*rsep) + rsep/2, 
                                       rsep), nommag)
        z_rows = np.multiply([-3.6e-3, -1.3e-3, 1.3e-3, 3.6e-3], nommag)
        # positions to scale
        rs = np.add(r_cols, self.view_radius)
        zs = np.add(z_rows, self.z_offset)
        
        # use scalings to locate centres, produce list of coordinates
        apdposns = []
        rrrs = []
        zzzs = []
        for j in np.arange(len(z_rows)):
            for i in np.arange(len(r_cols)):
                apdscaling = calc_view_geom(rs[i], zs[j])/calc_view_geom()
                rcoord = np.add(r_cols[i]*apdscaling, self.view_radius)
                zcoord = np.add(z_rows[j], self.z_offset)*apdscaling
                rrrs.append(rcoord)
                zzzs.append(zcoord)
                apdposns.append([rcoord, zcoord])
        
        self.apdpos = np.asarray(apdposns)  # the R,z positions of the BES array views
        
        self.cut_to_beam(ss_nbi_time, ss_nbi_data)
        
    def cut_to_beam(self, ss_nbi_time, ss_nbi_data):
        # BES data only expected or useful for the time where the beam is on
        # find the beam start and end times
        nbi_max = ss_nbi_data.max()
        expected_nbi = 0.9*nbi_max
        nbi_limit = 0.9*expected_nbi
        nbi_start = ss_nbi_time[np.argwhere(ss_nbi_data > nbi_limit)[0][0]]
        nbi_end = ss_nbi_time[np.argwhere(ss_nbi_data > nbi_limit)[-1][0]]
        nbi_startidx = np.argwhere(ss_nbi_data > nbi_limit)[0][0]
        nbi_endidx = np.argwhere(ss_nbi_data > nbi_limit)[-1][0]
        new_nbi_time = ss_nbi_time[nbi_startidx : nbi_endidx]
        new_nbi_power = ss_nbi_data[nbi_startidx : nbi_endidx]
        try:
            nbi_realstart = new_nbi_time[np.argwhere(new_nbi_power < 0.9*nbi_limit)[-1][0]]
        except:
            nbi_realstart = nbi_start
        start_time = (np.abs(self.time.raw - nbi_realstart)).argmin()
        end_time = (np.abs(self.time.raw - nbi_end)).argmin()
        self.time.cut = self.time.raw[start_time : end_time]
        self.data.cut = self.data.raw[:, start_time : end_time]
        self.nbi_times.first_beam_time = nbi_start
        self.nbi_times.real_start_time = nbi_realstart
        self.nbi_times.beam_end_time = nbi_end

    def calc_fluct_data(self, cutoff=600.0, sampling_freq=2.0e6):
        # use a low-pass filter to filter out some of the beam signal
        bes_fluct_data = []
        for i in range(self.nchan):
            zero_data = self.data.raw[i,:(np.abs(self.time.raw 
                                             - self.nbi_times.first_beam_time)).argmin()]
            # take average of this data and subtract from BES data to get it 
            # to start at zero
            zero_data_mean = np.mean(zero_data)
            shifted_data = self.data.cut[i, :] - zero_data_mean
            # to prevent the filter from mibehaving too much, subtract the 
            # mean of the signal before filtering
            mean_data = np.mean(shifted_data)
            data_for_filter = shifted_data - mean_data
            # sampling frequency of 2MHz for BES, cutoff for low-pass filter 
            # is 600Hz 
            beamsig = butter_lowpass_filter(data_for_filter, cutoff, 
                                            sampling_freq) + mean_data
            filtered_data = shifted_data/beamsig - 1.
            bes_fluct_data.append(filtered_data)
        self.data.fluct = np.asarray(bes_fluct_data)
    
    def calc_rms(self, window=100):
        # convolved r.m.s. fluctuation values
        weights = np.repeat(1.0, window)/window
    
        rms_data = []
        for i in range(self.nchan):
            fluct_squared = self.data.fluct[i]**2
            averaged_fluct = np.convolve(fluct_squared, weights, 'valid')
            rms_data.append(np.sqrt(averaged_fluct))
        rms_data = np.asarray(rms_data)
        
        self.time.rms = self.time.cut[int(window/2):-(int(window/2) - 1)]
        self.data.rms = rms_data
        
# adapted from AF, calculates the APD magnification based on distance 
# to beam axis
def calc_view_geom(view_radius=None, view_height=None):
    # View geometry parameters
    _r_beam = 0.7  # Beam tangency radius [m]
    _r_beam_port = 2.033  # Major radius of beam port [m]
    _dphi = scipy.deg2rad(30.)  # Angle between collection mirror M1 & NBI port
    _r_mirror = 1.869  # Major radius of collection mirror M1 [m]
    _z_mirror = 0.395  # Elevation of mirror centre below mid-plane
    _vr_nom = 1.2
    _z_nom = 0.
    # Initiate with nominal viewRadius if not specified
    if view_radius is None:
        calc_radius = _vr_nom
    else:
        calc_radius = view_radius

    # Initiate with nominal viewHeight (above midplane) if not specified
    if view_height is None:
        calc_height = _z_nom
    else:
        calc_height = view_height

    # Calculate geometry for specified viewRadius
    # Angle of viewed location from  beam tangency point [rad]
    alpha_view = np.arccos(_r_beam/calc_radius)
    # Angle between major radius to beam tangency point and beam port
    _beta = np.arccos(_r_beam/_r_beam_port)
    # Angle between radius of viewed location and radius of collection mirror
    delta = _beta - alpha_view + _dphi
    # Distance of viewed location from point above mirror in mid-plane
    d_view = np.sqrt(calc_radius**2 + _r_mirror**2 - 
                     2.0*calc_radius*_r_mirror*np.cos(delta))
    # Elevation angle of LoS above horizontal
    elev = np.arctan2((_z_mirror - calc_height), d_view)
    # Distance of viewed location from mirror M1
    l_view = d_view/np.cos(elev)

    return l_view


def get_bes_data(shot_number, view_radius=None, offset=None, 
                 nchan=32, spike_remover=False):
    """    RETURNS:
    bes_data   -   object containing channel ordered data and shot information. 

    The channel ordering follows the agreed convention for both the rotated 
    and unrotated arrays. 
    That is decreasing number from lower major radius to higher major radius 
    and then increasing from lower poloidal value to higher poloidal value,
    i.e. (Also channels start at +1)
    ^ increasing poloidal height
    |31, 30, 29, 28, 27, 26, 25, 24
    |23, 22, 21, 20, 19, 18, 17, 16
    |15, 14, 13, 12, 11, 10, 9, 8
    |7, 6, 5, 4, 3, 2, 1, 0
    ------> increasing major radius
    which I then correct to be:
    ^ increasing poloidal height
    |24, 25, 26, 27, 28, 29, 30, 31
    |16, 17, 18, 19, 20, 21, 22, 23
    |8, 9, 10, 11, 12, 13, 14, 15
    |0, 1, 2, 3, 4, 5, 6, 7
    ------> increasing major radius
    """
    data = []
    channel = np.arange(0, nchan) + 1
    for i in range(0, nchan):
        d = udaClient.get("xbt/channel"+str(channel[i]).zfill(2), shot_number)
        data.append(d.data)
    data = np.asarray(data)
    time_data = d.time.data
    
    # THERE ARE PROBLEMS WITH THIS!! WHY??
    # find viewradius in logs (in wiki)
    if view_radius is not None:
        view_radius = view_radius
    else:
        try:
            filename = '$MAST_DATA/{0:s}/LATEST/xbt{1:s}.nc'.format(
                    str(shot_number), str(shot_number).zfill(6))
            item = '/devices/d4_mirror/viewRadius'
            
            signal = udaClient.get(item, filename)
            view_radius = signal.data[0]/1.
        except:
            print("View radius data not available. Setting view radius to 1.2")
            view_radius = 1.2

    # get Z coordinate of view centre
    if offset is not None:
        z = offset
    else:
        z = 0.0
    
    sprm_data = []
    if spike_remover==True:
        pulse_num = []
        for i in range(0, nchan):
            signal, npulses = sr.spike_filter(data[i], limit=5e-3)
            sprm_data.append(signal)
            pulse_num.append(npulses)
        data = np.array(sprm_data)
        print(pulse_num)
      
    # relabel data channels to be increasing radially outwards 
    # (like the database!)
    new_data = np.zeros(np.shape(data))
    old = [31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 
           15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    new = [24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23, 
           8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]
    for i, old_chan in enumerate(old):
        new_data[new[i], :] = data[old_chan, :]
    
    # get SS beam data
    ss_nbi = udaClient.get('ANB_SS_SUM_POWER', shot_number)
    
    # Create the bes data object
    bes_object = BESClass(shot_number, view_radius, z, time_data, new_data, 
                          ss_nbi.time.data, ss_nbi.data)

    return bes_object
