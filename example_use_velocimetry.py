from imports import *
from BESClass import *
from functions_BES import *
from cctde_velocimetry import *

# example use for shot 26554 L-H transition
shot = 26554
timeslice = [0.235, 0.255]

# get information on view radius from file
view_radii = np.genfromtxt('BES_viewradius.csv', delimiter=',')
view_radius = None
for item in view_radii:
    if int(item[0]) == shot:
        view_radius = item[1]
# set up bes data object
bes_data = get_bes_data(shot, view_radius=view_radius, spike_remover=True)

# calculate fluctuation data
bes_data.calc_fluct_data()

# calculate radial and poloidal velocities with correlation values and errors 

# number of points to include in cross-correlation 
# (can change this, e.g. 200 or 1000) kind of like a signal-to-noise ratio
xcorr_length = 500
# e.g. using 20 percent of xcorr length (+/- 10 percent)
fit_range = 0.2

row_idxs = list(range(bes_data.nrows))
col_idxs = list(range(bes_data.ncols))

# this example uses a Gaussian fit to find the time delays and doesn't remove 
# the poloidal average
# if you want to test code quickly, add arguments 
# cols_to_calc=[2], rows_to_calc=[1] 
# to calculate only channel 10 
results_dict = calc_cctde_velocimetry(
    bes_data, timeslice, xcorr_length, bandpass_filter='automatic', 
    fit_range=fit_range, calc_poloidal=True, calc_radial=True, 
    remove_poloidal_average=False, use_gaussian_fit=True)

# save the data to netCDF files as calculations take some time
# can choose your own filename
file_name = 'shot' + str(shot) + '_bpf_xcorr500_fit20pc_'

pol_vels_xr = xr.DataArray(results_dict["poloidal"]["velocities"], 
                          dims=('time', 'row', 'column'), 
    coords={'time':results_dict["poloidal"]["mid_time_points"], 
            'row':row_idxs, 'column':col_idxs})
pol_vels_xr.to_netcdf(file_name + 'pol_vels.nc')

pol_corr_xr = xr.DataArray(results_dict["poloidal"]["correlations"], 
                            dims=('time', 'row', 'column'), 
    coords={'time':results_dict["poloidal"]["mid_time_points"], 
            'row':row_idxs, 'column':col_idxs})
pol_corr_xr.to_netcdf(file_name + 'pol_corr.nc')

pol_err_xr = xr.DataArray(results_dict["poloidal"]["errors"], 
                          dims=('time', 'row', 'column'), 
    coords={'time':results_dict["poloidal"]["mid_time_points"], 
            'row':row_idxs, 'column':col_idxs})
pol_err_xr.to_netcdf(file_name + 'pol_err.nc')

rad_vels_xr = xr.DataArray(results_dict["radial"]["velocities"], 
                          dims=('time', 'row', 'column'), 
    coords={'time':results_dict["radial"]["mid_time_points"], 
            'row':row_idxs, 'column':col_idxs})
rad_vels_xr.to_netcdf(file_name + 'rad_vels.nc')

rad_corr_xr = xr.DataArray(results_dict["radial"]["correlations"], 
                            dims=('time', 'row', 'column'), 
    coords={'time':results_dict["radial"]["mid_time_points"], 
            'row':row_idxs, 'column':col_idxs})
rad_corr_xr.to_netcdf(file_name + 'rad_corr.nc')

rad_err_xr = xr.DataArray(results_dict["radial"]["errors"], 
                          dims=('time', 'row', 'column'), 
    coords={'time':results_dict["radial"]["mid_time_points"], 
            'row':row_idxs, 'column':col_idxs})
rad_err_xr.to_netcdf(file_name + 'rad_err.nc')
