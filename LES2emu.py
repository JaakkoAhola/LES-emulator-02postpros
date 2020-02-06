# Functions extracting emulator and any other data from LES output NetCDF files,
# and collection of functions for generating LES inputs.
#
#	Tomi Raatikanen 18.1.2019
#
# Functions
# =========
# Use Python import to make these functions available, e.g. from LES2emu import GetEmuVars, get_netcdf_variable
#
# a) Functions for extracting data from the LES outputs
#	GetEmu2Vars(path)
#	GetEmu1Vars(fname,tstart,tend,[ttol,start_offset,end_offset])
#	get_netcdf_variable(fname,var_name,target_time,[end_time])
#	extract_write_data(fname_template,specs,[name_out,nmax])
#	get_netcdf_updraft(fname,tstart,tend,[ttol,tol_clw])
#
# b) Functions for generating LES inputs
#	calc_cloud_base(p_surf,theta,rw)
#	calc_lwc_altitude(p_surf,theta,rw,zz)
#	solve_rw(p_surf,theta,lwc,zz)
#	solve_rw_lwp(p_surf,theta,lwp,pblh)
#
# c) Helper functions
#	calc_psat_w(T)
#	calc_sat_mixr(p,T)
#	calc_rh(rw,T,press)
#
# Notes
# =====
# 1) Input file name should contain complete path in addition to the file name
# 	e.g. '/ibrix/arch/ClimRes/aholaj/case_emulator_DESIGN_v1.4.0_LES_cray.dev20170324_LVL4/emul01/emul01.ts.nc'
# 2) Voima requires python 2.7.10, so execute "module load Python/2.7.10"
#


def GetEmu1Vars(fname,tstart,tend,ttol=3600.,start_offset=0,end_offset=0):
	# Function calculates LES output variables for emulator v1.0 as defined in the ECLAIR proof-of-concept document
	#	https://docs.google.com/document/d/1L-YyJLhtmLYg4rJYo5biOW96eeRC7z_trZsow_8TbeE/edit
	# Inputs:
	# 	fname			Complete path and name of a time statistics file (*.ts.nc)
	#	start, tend	Time window (s)
	# Optional inputs
	#	ttol			Time tolelance (s) for finding averaging window
	#	start_offset	Point offset to start index (time in the NetCDF files is the save time, so typically should ignore the first points)
	#	end_offset	-||- end ind index
	#
	# Example:
	#	file='/ibrix/arch/ClimRes/aholaj/case_emulator_DESIGN_v1.4.0_LES_cray.dev20170324_LVL4/emul01/emul01.ts.nc'
	#	tstart=2.5*3600
	#	tend=3.5*3600
	#	cfrac, CDNC, prcp, dn, we, cfrac_std, CDNC_std, prcp_std, dn_std, we_std = GetEmu1Vars(file,tstart,ttol=10.,start_offset=1)
	#
	import os
	import netCDF4 as netcdf
	import numpy
	#
	# Outputs
	cfrac=-999.	# Cloud fraction
	CDNC=-999.	# Cloud droplet number concentration in cloudy columns (#/kg)
	prcp=-999.		# Precipitation tendency = domain mean surface precipitation (kg/m^2/s)
	dn=-999.		# In-cloud aerosol number loss tendency = change in interstitial aerosol+cloud droplet number concentration in cloudy columns (#/kg/s)
	we=-999.		# Mixing between FT and BL = domeain mean entrainment velocity (m/s)
	#
	# ... and their standard deviations
	cfrac_std=-999.; CDNC_std=-999.; prcp_std=-999.; dn_std=-999.; we_std=-999.
	#
	# File must exist
	if not os.path.lexists(fname):
		print (fname+' not found!')
		return cfrac, CDNC, prcp, dn, we, cfrac_std, CDNC_std, prcp_std, dn_std, we_std,
	#
	# Open the target NetCDF file
	ncid = netcdf.Dataset(fname,'r')
	#
	# Time
	# ====
	if 'time' not in ncid.variables:
		print  ('Time not found from '+fname+'!')
		return cfrac, CDNC, prcp, dn, we, cfrac_std, CDNC_std, prcp_std, dn_std, we_std,
	times = ncid.variables['time']
	#
	# Find the closest matching time points
	ind_tstart=0
	ind_tend=0
	i=0
	for t in times:
		if abs(t-tstart)<abs(times[ind_tstart]-tstart): ind_tstart=i
		if abs(t-tend)<abs(times[ind_tend]-tend): ind_tend=i
		i+=1
	#
	if abs(times[ind_tstart]-tstart)>ttol or abs(times[ind_tend]-tend)>ttol:
		print  ('Matching start or end time not found from '+fname+'!')
		return cfrac, CDNC, prcp, dn, we, cfrac_std, CDNC_std, prcp_std, dn_std, we_std,
	#
	# Apply offset (typically the first point is ignored
	ind_tstart+=start_offset
	ind_tend+=end_offset
	if ind_tstart<0 or ind_tstart>ind_tend or ind_tend>=len(times):
		print  ('Invalid data range for '+fname+': ',ind_tstart,ind_tend,len(times))
		return cfrac, CDNC, prcp, dn, we, cfrac_std, CDNC_std, prcp_std, dn_std, we_std
	#
	#
	# Outputs
	# ========
	# Cloud fraction
	if 'cfrac' not in ncid.variables:
		print  ('Cloud fraction not found from '+fname+'!')
		return cfrac, CDNC, prcp, dn, we, cfrac_std, CDNC_std, prcp_std, dn_std, we_std,
	#
	# Need cloud fractions for normalizing domain mean interstitial and cloud droplet number concentrations
	cfrac_ts=ncid.variables['cfrac'][ind_tstart:ind_tend]
	ncfrac = sum( cfrac_ts>0.0 ) # The number of non-zero cloud fractions
	#
	cfrac = numpy.mean( cfrac_ts )
	cfrac_std = numpy.std( cfrac_ts )
	#
	if 'Nc_ic' in ncid.variables:	# Level 4 = SALSA microphysics
		# Cloud droplet number concentration averaged over cloudy columns (#/kg)
		CDNC,CDNC_std=average_scaled(ncid.variables['Nc_ic'][ind_tstart:ind_tend],cfrac_ts)
		#
		# Surface precipitation (kg/m^2/s)
		if ind_tstart < ind_tend:
			prcp = numpy.mean( ncid.variables['rmH2Opr'][ind_tstart:ind_tend] )
			prcp_std = numpy.std( ncid.variables['rmH2Opr'][ind_tstart:ind_tend] )
		else:
			prcp = ncid.variables['rmH2Opr'][ind_tstart]
			prcp_std = -999.
		#
		# Change in in-cloud aerosol+cloud droplet number concentration
		if ncfrac>=2: # Linear fit needs at least two data points
			tt = ncid.variables['time'][ind_tstart:ind_tend] 	# Time (s) vector
			nc = ncid.variables['Nc_ic'][ind_tstart:ind_tend]	# Cloud droplets (domain mean)
			nc += ncid.variables['Na_int'][ind_tstart:ind_tend]	# + interstitial aerosol (domain mean)
			# Normalize by cloud fraction => concentrations for cloudy columns
			i=0
			for cf in cfrac_ts:
				if cf>0:
					nc[i]/=cf
				else:
					# Divide-by-zero => NaN
					nc[i]=float('nan')
				i+=1
			#
			a,dn,a_std,dn_std=ls_fit(tt,nc)	# Least squares fit (nc=a+b*tt)
		else:
			dn=-999.
			dn_std=-999.
	else:		# Level 3 = saturation adjustment method (given CDNC)
		# Cloud droplet number concentration (#/kg): fixed
		if ind_tstart < ind_tend:
			CDNC = numpy.mean( ncid.variables['CCN'][ind_tstart:ind_tend] )
			CDNC_std = numpy.std( ncid.variables['CCN'][ind_tstart:ind_tend] )
		else:
			CDNC = ncid.variables['CCN'][ind_tstart]
			CDNC_std = -999.
		#
		# Surface precipitation (kg/m^2/s): variable prcp is in W/m^2=J/s/m^2, which can be
		# converted to kg using latent heat of vaporization (2.5e+06 J/kg)
		if ind_tstart < ind_tend:
			prcp = numpy.mean( ncid.variables['prcp'][ind_tstart:ind_tend] )/2.5e6
			prcp_std = numpy.std( ncid.variables['prcp'][ind_tstart:ind_tend] )/2.5e6
		else:
			prcp = ncid.variables['prcp'][ind_tstart]/2.5e6
			prcp_std = -999.
		#
		# Change in in-cloud aerosol+cloud droplet number concentration: N/A
	#
	# Entrainment velocity (m/s)
	#	we=dz/dt+D*z, where z is PBL height and D is large scale divergence (1.5e-6 1/s) (see e.g. Kazil et al., ACP, 2016).
	if ind_tstart < ind_tend:
		# Must have at least two points for the slope, but should haev more than that
		zz = ncid.variables['zi1_bar'][ind_tstart:ind_tend]	# PBL height (m) vector
		tt = ncid.variables['time'][ind_tstart:ind_tend]	# Time (s) vector
		a,dzdt,a_std,dzdt_std=ls_fit(tt,zz)	# Least squares fit (zz=a+b*tt)
		z=numpy.mean(zz)	# Mean PBL height
		we=dzdt+1.5e-6*z
		we_std=dzdt_std
	else:
		we = -999.
		we_std = -999.
	#
	# Close file
	ncid.close()
	#
	# All done
	return cfrac, CDNC, prcp, dn, we, cfrac_std, CDNC_std, prcp_std, dn_std, we_std,


def get_netcdf_updraft(fname,tstart,tend,ttol=3600.,tol_clw=1e-5):
	# Function for calculating mean positive updraft velocities and cloud droplet number concentrations
	# at cloud base (see Romakkaniemi et al., 2009) from 4D data (*.nc).
	#
	# Romakkaniemi, S., G. McFiggans, K.N. Bower, P. Brown, H. Coe, T.W. Choularton, A comparison between
	# trajectory ensemble and adiabatic parcel modelled cloud properties and evaluation against airborne
	# measurements, J. Geophys. Res., doi:10.1029/2008JD011286, 2009
	#
	# Inputs:
	#	fname			Complete file path and name (*.nc)
	#	start, tend	Averaging time window (s)
	# Optional inputs
	#	ttol			Time tolelance (s) for finding averaging window
	#	tol_clw		Cloud liquid water mixing ratio (kg/kg) for the cloud base
	#
	# Example:
	#	file='/arch/eclair/UCLALES-SALSA_training_simulations/case_emulator_DESIGN_v3.0.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3/emul001/emul001.nc'
	#	w,cdnc,cdnc_w,n=get_netcdf_updraft(file,9000.,12600.,ttol=10.)
	#	print w,cdnc,cdnc_w,n
	#
	import os
	import netCDF4 as netcdf
	import numpy
	#
	# File must exist
	if not os.path.lexists(fname):
		raise RuntimeError(fname+' not found!')
	#
	# Open the target NetCDF file
	ncid = netcdf.Dataset(fname,'r')
	#
	if 'time' not in ncid.variables:
		raise RuntimeError('Time not found from '+fname+'!')
	elif 'w' not in ncid.variables:
		raise RuntimeError('Variable w not found from '+fname+'!')
	elif 'l' not in ncid.variables:
		raise RuntimeError('Variable l not found from '+fname+'!')
	elif 'time' not in ncid.variables['w'].dimensions or 'time' not in ncid.variables['l'].dimensions:
		raise RuntimeError('Time is not a dimension for w or l (file '+fname+')!')
	#
	# Time
	times = ncid.variables['time']
	#
	# Dimensions
	dims=ncid.variables['l'][0,].shape #  x, y, z
	#
	# Optional: CDNC from UCLALES-SALSA simulations
	cdnc_calc='S_Nc' in ncid.variables
	#
	#
	# Outputs
	wpos=0.		# Mean positive updraft velocity at the cloud base (m/s)
	w2pos=0.		# Velocity weighted mean positive updraft velocity at the cloud base (m/s)
	cdnc_p=0.		# Mean cloud droplet number concentration at the cloud base with positive updraft velocity (1/kg)
	cdnc_wp=0.	# Velocity weigted mean cloud droplet number concentration at the cloud base with positive updraft velocity (1/kg)
	n=0				# Number of cloud bases with positive updraft (-)
	#
	ind=-1
	for tt in times:
		# Time range
		ind+=1
		if tt<tstart-ttol:
			continue
		elif tt>tend+ttol:
			break # Assuming monotonic time
		#
		# Data
		w=ncid.variables['w'][ind,]
		l=ncid.variables['l'][ind,]
		if cdnc_calc: cdnc=ncid.variables['S_Nc'][ind,]
		#
		# Calculations
		if True:
			# This is much faster
			for i in range(0,dims[0]):
				for j in range(0,dims[1]):
					kk, = numpy.where(l[i,j,:]>tol_clw)
					if len(kk)>0 and w[i,j,kk[0]]>0.:
						k=kk[0]
						n+=1
						wpos+=w[i,j,k]
						w2pos+=w[i,j,k]**2
						if cdnc_calc:
							cdnc_p+=cdnc[i,j,k]
							cdnc_wp+=w[i,j,k]*cdnc[i,j,k]
		else:
			# The old method
			i=0; j=0
			while i<dims[0] and j<dims[1]:
				k=0
				while k<dims[2]:
					if l[i,j,k]>tol_clw:
						# Found cloud base, but only the positive updraft velocities are counted
						if w[i,j,k]>0.:
							n+=1
							wpos+=w[i,j,k]
							w2pos+=w[i,j,k]**2
							if cdnc_calc:
								cdnc_p+=cdnc[i,j,k]
								cdnc_wp+=w[i,j,k]*cdnc[i,j,k]
						break
					k+=1
				if j+1<dims[1]:
					j+=1
				else:
					i+=1
					j=0
	#
	if n>0:
		w2pos/=wpos
		wpos/=n
		if cdnc_calc:
			cdnc_p/=n
			cdnc_wp/=(wpos*n)
		else:
			cdnc_p=-999.
			cdnc_wp=-999.
	else:
		wpos=-999.
		w2pos=-999.
		cdnc_p=-999.
		cdnc_wp=-999.
	#
	# Close file
	ncid.close()
	#
	# Outputs: mean positive updraft velocity and cloud droplet number concentrations (mean and weighted with velocity) at the cloud base
	return wpos,w2pos,cdnc_p,cdnc_wp,n




def get_netcdf_variable(fname,var_name,start_time,end_time=-10000.,ttol=3600.,start_offset=0,end_offset=0):
	# Function for extracting data from a NetCDF file based on the given time value (or range).
	#
	# Inputs:
	#	fname			Complete file path and name
	#	var_name		NetCDF variable name
	#	start_time	Target or start (when end_time is specified) time value
	# Optional inputs
	#	end_time		Optional end time value
	#	ttol			Time tolelance (s) for finding averaging window
	#	start_offset	Point offset to start index (time in the NetCDF files is the save time, so typically should ignore the first points)
	#	end_offset	-||- end index
	#
	# Example:
	#	file='/ibrix/arch/ClimRes/aholaj/case_emulator_DESIGN_v1.4.0_LES_cray.dev20170324_LVL4/emul01/emul01.ts.nc'
	#	lmax=get_netcdf_variable(file,'lmax',3*3600,ttol=10)
	#	lmax=get_netcdf_variable(file,'lmax',2.5*3600,3.5*3600,ttol=10.,start_offset=1)
	import os	
	import numpy
	import netCDF4 as netcdf
	#
	# File must exist
	if not os.path.lexists(fname): raise RuntimeError(fname+' not found!')
	#
	# Open the target NetCDF file
	ncid = netcdf.Dataset(fname,'r')
	#
	if 'time' not in ncid.variables:
		raise RuntimeError('Time not found from '+fname+'!')
	elif var_name not in ncid.variables:
		raise RuntimeError('Variable '+var_name+' not found from '+fname+'!')
	elif 'time' not in ncid.variables[var_name].dimensions:
		raise RuntimeError('Time is not a dimension for '+var_name+' (file '+fname+')!')
	#
	# Time
	times = ncid.variables['time']
	#
	# Find the correct time value
	i=0
	if end_time>-9999.:
		# Time from start_time to end_time (closests matching start and end time values)
		ind_start=0
		ind_end=0
		for tt in times:
			# Closest match
			if abs(tt-start_time)<abs(times[ind_start]-start_time): ind_start=i
			if abs(tt-end_time)<abs(times[ind_end]-end_time): ind_end=i
			i+=1
		#
		if abs(times[ind_start]-start_time)>ttol or abs(times[ind_end]-end_time)>ttol:
			print ('Matching start or end time not found from '+fname+'!')
			return -999.
		#
		# Apply offset (typically the first point is ignored
		ind_start+=start_offset
		ind_end+=end_offset
		if ind_start<0 or ind_start>ind_end or ind_end>=len(times):
			print ('Invalid data range for '+fname+'!')
			return -999.
		#
		# Average over time dimension
		ind=ncid.variables[var_name].dimensions.index('time')
		#
		out=numpy.mean( ncid.variables[var_name][ind_start:ind_end,],axis=ind )
		# Could add standard deviations?
		#out_std = numpy.std( ncid.variables[var_name][ind_start:ind_end,],axis=ind )
	else:
		# Single time value (closest match)
		ind=0
		for tt in times:
			# Closest match
			if abs(tt-start_time)<abs(times[ind]-start_time): ind=i
			i=i+1
		#
		if abs(times[ind]-tstart)>ttol:
			print ('Matching time not found from '+fname+'!')
			return -999.
		#
		# Apply offset (typically the first point is ignored
		ind+=start_offset
		if ind<0 or ind>=len(times):
			print ('Invalid index for '+fname+'!')
			return -999.
		#
		out=ncid.variables[var_name][ind,]
	#
	# Close file
	ncid.close()
	return out


def GetEmu2Vars(path):
	# Function calculates LES output variables for emulator v2.0 as defined in the ECLAIR proof-of-concept document
	#	https://docs.google.com/document/d/1L-YyJLhtmLYg4rJYo5biOW96eeRC7z_trZsow_8TbeE/edit
	#
	# Inputs:
	# 	path			Complete path the root data
	# Outputs:
	#	A 2D array of values as described below
	#
	# Example:
	# path='/arch/eclair/UCLALES-SALSA_training_simulations/case_emulator_DESIGN_v3.0.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3/'
	# data=GetEmu2Vars(path)
	#
	import os
	#
	# Time window
	tstart=2.5*3600
	tend=3.5*3600
	#
	if path.endswith('/'):
		fmt='emul%03u/emul%03u'
	else:
		fmt='/emul%03u/emul%03u'
	#
	# Examine the data files
	out=[]
	i=1
	while True:
		# Data must exist
		if not os.path.lexists( (path+fmt+'.nc')%(i,i) ) and not os.path.lexists( (path+fmt+'.ts.nc')%(i,i) ):
			if i==1:
				raise RuntimeError('Data not found from '+path+'!')
			else:
				break
		#
		# Emulator v2.0 variables
		#	- Tolerance for the start and end times is +/- 10 s
		#	- No need to ignore the first point when averaging instantaneous variables (rain processes and w calculated form 4D data,)
		#
		# Data file
		file=(path+fmt+'.ts.nc')%(i,i)
		# 1a) Rain water loss (evaporation + surface precipitation)
		# Change in column rain water due to condensation (kg/m^2/s)
		cond=get_netcdf_variable(file,'cond_rr',tstart,tend,ttol=10.)
		# Change in column rain water due to sedimentation (kg/m^2/s)
		sedi=get_netcdf_variable(file,'sedi_rr',tstart,tend,ttol=10.)
		#
		# 1b) Rain water production (not including divergence - with that the total production is the same as total loss)
		# Change in column rain water due to coagulation (kg/m^2/s)
		coag=get_netcdf_variable(file,'coag_rr',tstart,tend,ttol=10.)
		# Change in column rain water due to autoconversion (kg/m^2/s)
		auto=get_netcdf_variable(file,'auto_rr',tstart,tend,ttol=10.)
		# Change in column rain water due to diagnostics (kg/m^2/s)
		diag=get_netcdf_variable(file,'diag_rr',tstart,tend,ttol=10.)
		#
		# 2) Precipitation statistics (ignore the first point, which is an everage from the previous time period)
		# Surface precipitation rate (W/m^2)
		prcp=get_netcdf_variable(file,'prcp',tstart,tend,ttol=10.,start_offset=1)
		# 1 W = J/s, which can be converted to mass flux by using latent heat of water (2.5e+6 J/kg)
		prcp/=2.5e6 # kg/m^2/s
		#
		# 3) Cloud base positive updraft velocity (m/s)
		file_4d=(path+fmt+'.nc') % (i,i)
		wpos,w2pos,cdnc_p,cdnc_wp,n = get_netcdf_updraft(file_4d,tstart,tend,ttol=10.)
		#
		out.append([i,cond,sedi,coag,auto,diag,prcp,wpos,w2pos,cdnc_p,cdnc_wp,n])
		#
		if i==1: print ('id        cond        sedi       coag       auto       diag       prcp    wpos   w2pos      cdnc_p     cdnc_wp        n')
		print ('%2g  %8.3e  %8.3e  %8.3e  %8.3e  %8.3e  %8.3e  %6.4f  %6.4f  %7.3e  %7.3e  %7g' % (i,cond,sedi,coag,auto,diag,prcp,wpos,w2pos,cdnc_p,cdnc_wp,n))
		#
		i+=1
	#
	# Output lines are: id, cond, sedi, coag, auto, diag, prcp, wpos, w2pos, cdnc_p, cdnc_wp, n
	return out


def extract_write_data(fname_template,specs,name_out='',nmax=9999,skip_errs=False):
	# Extract and process data from one or more NetCDF files, and write it to a text file (optional)
	#
	# Inputs:
	#	fname_template	File name template with complete path
	#						e.g. '/ibrix/arch/ClimRes/aholaj/case_emulator_DESIGN_v1.4.0_LES_cray.dev20170324_LVL4/emul%02u/emul%02u.ts.nc'
	#	specs				List of variables including slizing and numpy operations
	#	name_out			Output file name (optional)
	# 	nmax				Maximum number of files (optional)
	#	skip_errs			Don't stop on errors - needed when complete data set is not available (saves just NaN)
	#
	# Examples:
	#	fname_template='/ibrix/arch/ClimRes/aholaj/case_emulator_DESIGN_v1.4.0_LES_cray.dev20170324_LVL4/emul%02u/emul%02u.ts.nc'
	#	specs=['cfrac[10]','wmax[10]','lwp_bar[10]']
	#	aa=extract_write_data(fname_template,specs,name_out='tmp.dat')
	#	specs=['cfrac']
	#	aa=extract_write_data(fname_template,specs,name_out='tmp.dat')
	#
	#	fname_template='/ibrix/arch/ClimRes/aholaj/case_emulator_DESIGN_v1.4.0_LES_cray.dev20170324_LVL4/emul%02u/emul%02u.ps.nc'
	#	specs=['numpy.amax(l[10,:])']
	#	aa=extract_write_data(fname_template,specs,name_out='tmp.dat')
	#
	#	fname_template='/ibrix/arch/ClimRes/aholaj/case_emulator_DESIGN_v1.4.0_LES_cray.dev20170324_LVL4/emul%02u/emul%02u.nc'
	#	specs=['numpy.amax(numpy.amax(numpy.amax(l[2,:],axis=0),axis=0),axis=0)']
	#	aa=extract_write_data(fname_template,specs,name_out='tmp.dat')
	#
	import os
	import netCDF4 as netcdf
	import numpy
	import sys
	#
	# Function for converting command line commands to NetCDF format
	def interpret_fun(cmd):
		# Interpret function call, e.g. 'numpy.amax(l[89,:])': just replace variable name x with "ncid.variables['x']",
		# e.g. 'numpy.amax(l[2,100,100,:])', numpy.amax(l[89,:])',numpy.amax(l)' or 'numpy.amax(P_Rwca,axis=0)'
		# Now also function calls like "thl[numpy.abs(ncid.variables['time'][:]-10800.).argmin(),:]" are accepted!
		frst=-1
		lst=-1
		i=0
		for tt in cmd:
			if (tt=='[' or tt==')' or tt==',') and lst==-1:
				# e.g. 'numpy.amax(l[89,:])', 'numpy.amax(l)', 'numpy.amax(P_Rwca,axis=0)' or thl[numpy.abs(ncid.variables['time'][:]-10800.).argmin(),:]"
				lst=i
				if frst==0:
					fun='ncid.variables[\''+cmd[frst:lst]+'\']'+cmd[lst:]
				else:
					fun=cmd[:frst+1]+'ncid.variables[\''+cmd[frst+1:lst]+'\']'+cmd[lst:]
				return fun
			elif tt=='(' or i==0:
				frst=i
			i+=1
		# No solution
		return cmd
		#
		# The old version
		frst=-1
		lst=-1
		i=0
		for tt in cmd:
			if (tt=='[' or tt==')' or tt==',') and lst==-1:
				# e.g. 'numpy.amax(l[89,:])', 'numpy.amax(l)' or 'numpy.amax(P_Rwca,axis=0)'
				lst=i
			elif tt=='(':
				frst=i
			i+=1
		# Return complete command as fun
		fun=cmd[:frst+1]+'ncid.variables[\''+cmd[frst+1:lst]+'\']'+cmd[lst:]
		return fun
	#
	#
	# Any '%' in file name template will be replaced by counter i=1,2,3,...
	n=fname_template.count('%')
	if n==0: nmax=1	# Template as is
	#
	# Output to text file
	if len(name_out): fid_out=open(name_out,'w')
	#
	nerr=0
	files=0		# Count files
	values=0	# Count values extracted
	out=[] 		# Complete output
	for i in range(1,nmax):
		# File name with full path
		if n==0:
			file_name=fname_template
		elif n==1:
			file_name=fname_template % (i)
		elif n==2:
			file_name=fname_template % (i,i)
		elif n==3:
			file_name=fname_template % (i,i,i)
		else:
			# No more than three?
			file_name=fname_template % (i,i,i,i)
		#
		ncid=0
		if not os.path.isfile(file_name):
			if i==1 and n>0: print (file_name+' not found!')
			if not skip_errs or i>90:
				break
			else:
				# Ignore missing file (<90)
				ncid=-999
				nerr+=1
				msg=file_name+' not found!'
				#
				#
				#row=[] # one row
				#for nam in specs:
				#	row.append(obj)
				#
				#out.append(row)
				#
				# Save data
				#if len(name_out):
				#	# Save variables as space separated strings
				#	if not hasattr(obj, '__iter__'):
				#		# Scalar
				#		fid_out.write( str(obj)+" " )
				#		values+=1
				#	else:
				#		# Vector
				#		for val in obj:
				#			fid_out.write( str(val)+" " )
				#			values+=1
				#
				#continue
		#
		# Open input file
		if ncid==0: ncid = netcdf.Dataset(file_name,'r')
		#
		# Outputs
		row=[] # one row
		for nam in specs:
			# Interpret command
			if ncid<0:
				# File not found
				obj = float('nan')	# Scalar works quite often
			elif '(' in nam:
				# There is a call to a function
				fun=interpret_fun(nam)
				try:
					obj =eval( fun )
				except:
					if not skip_errs:
						print  ("Unexpected error:", sys.exc_info()[0])
						raise
					#
					# Ignore errors
					obj = float('nan')	# Scalar works quite often
					nerr+=1
					msg=sys.exc_info()[0]
			elif '[' in nam:
				# Selected data range
				name=nam[:nam.index('[')]
				ind=nam[nam.index('['):]
				try:
					obj =eval( 'ncid.variables[\''+name+'\']'+ind )
				except:
					if not skip_errs:
						print  ("Unexpected error:", sys.exc_info()[0])
						raise
					#
					# Ignore errors
					obj = float('nan')	# Scalar works quite often
					nerr+=1
					msg=sys.exc_info()[0]
			else:
				# Data as is
				try:
					obj = ncid.variables[nam][:]
				except:
					if not skip_errs:
						print  ("Unexpected error:", sys.exc_info()[0])
						raise
					#
					# Ignore errors
					obj = float('nan')	# Scalar works quite often
					nerr+=1
					msg=sys.exc_info()[0]
			#
			# Append data
			row.append(obj)
			#
			# Save data
			if len(name_out):
				# Save variables as space separated strings
				if not hasattr(obj, '__iter__'):
					# Scalar
					fid_out.write( str(obj)+" " )
					values+=1
				else:
					# Vector/matrix
					for val in obj:
						if not hasattr(val, '__iter__'):
							# Scalar (1D vector)
							fid_out.write( str(val)+" " )
							values+=1
						else:
							# Vector (2D matrix)
							for val2 in val:
								fid_out.write( str(val2)+" " )
								values+=1
		#
		# New line
		if len(name_out):	fid_out.write( "\r\n" )
		#
		out.append(row)
		#
		if ncid>0: ncid.close()
		files+=1
	#
	if len(name_out):
		fid_out.close()
		print (str(files)+' files examined, '+str(values)+' values saved to '+name_out)
		if nerr>0: print  ('   '+str(nerr)+' error(s) ignored: ',msg)
	#
	# Return the data
	return out


#
# LES inputs and outputs
#

def calc_cloud_base(p_surf,theta,rw):
	# Calulate cloud base heigh when liquid water potential temperature (theta [K]) and water
	# vapor mixing ratio (rw [kg/kg]) are constants. Surface pressure p_surf is given in Pa.
	# For more information, see "lifted condensation level" (LCL).
	#
	# Constants
	R=287.04	# Specific gas constant for dry air (R_specific=R/M), J/kg/K
	Rm=461.5	# -||- for water
	ep2=Rm/R-1.0 #M_air/M_water-1
	cp=1005.0	# Specific heat for a constant pressure
	rcp=R/cp
	cpr=cp/R
	g=9.8
	p00=1.0e+05
	#
	# Integrate to cloud base altitude
	dz=1.			# 1 m resolution
	z=0.				# The first altitude
	press=p_surf	# Start from surface
	RH=0
	while RH<100 and z<10000:
		# Temperature (K)
		tavg=theta*(press/p00)**rcp
		#
		# Current RH (%)
		RH=calc_rh(rw,tavg,press)
		if RH>100: break
		#
		# From z to z+dz
		z+=dz
		# Virtual temperature: T_virtual=T*(1+ep2*rl)
		xsi=(1+ep2*rw)
		# Pressure (Pa)
		press-=g*dz*press/(R*tavg*xsi)
	#
	# No cloud
	if RH<100: return -999
	#
	# Return cloud base altitude
	return z


def calc_lwc_altitude(p_surf,theta,rw,zz):
	# Calculate cloud water mixing ratio at a given altitude z (m) when liquid water potential 
	# temperature (theta [k]) and water vapor mixing ratio (rw [kg/kg]) are constants. 
	# Surface pressure p_surf is given in Pa.
	#
	# Constants
	R=287.04	# Specific gas constant for dry air (R_specific=R/M), J/kg/K
	Rm=461.5	# -||- for water
	ep2=Rm/R-1.0 #M_air/M_water-1
	cp=1005.0	# Specific heat for a constant pressure
	rcp=R/cp
	cpr=cp/R
	g=9.8
	p00=1.0e+05
	alvl = 2.5e+06 #  ! latent heat of vaporization
	#
	# a) Integrate to cloud base altitude
	dz=1.			# 1 m resolution
	z=0.				# The first altitude
	press=p_surf	# Start from surface
	RH=0
	while z<zz:
		# Temperature (K) 
		tavg=theta*(press/p00)**rcp
		#
		# Current RH (%)
		RH=calc_rh(rw,tavg,press)
		if RH>100: break
		#
		# From z to z+dz
		z+=dz
		# Virtual temperature: T_virtual=T*(1+ep2*rl)
		xsi=(1+ep2*rw)
		# Pressure (Pa)
		press-=g*dz*press/(R*tavg*xsi)
	#
	# No cloud or cloud water
	if RH<100: return 0.0
	#
	# b) Integrate up to given altitude
	while z<zz:
		# From z to z+dz
		z+=dz
		#
		# Moist adiabatic lapse rate
		q_sat=calc_sat_mixr(press,tavg)
		tavg-=g*(1+alvl*q_sat/(R*tavg))/(cp+alvl**2*q_sat/(Rm*tavg**2))*dz
		#
		# New pressure
		xsi=(1+ep2*q_sat)
		press-=g*dz*press/(R*tavg*xsi)
	#
	# Return cloud water mixing ratio = totol - vapor
	return rw-q_sat


def solve_rw(p_surf,theta,lwc,zz):
	# Solve total water mixing ratio (rw, kg/kg) from surface pressure (p_surf, Pa), liquid water potential
	# temperature (theta, K) and liquid water mixing ratio (lwc) at altitude zz (m)
	#
	# Constants
	R=287.04	# Specific gas constant for dry air (R_specific=R/M), J/kg/K
	Rm=461.5	# -||- for water
	ep2=Rm/R-1.0 #M_air/M_water-1
	cp=1005.0	# Specific heat for a constant pressure
	rcp=R/cp
	cpr=cp/R
	g=9.8
	p00=1.0e+05
	alvl = 2.5e+06 #  ! latent heat of vaporization
	#
	# Mimimum water vapor mixing ratio is at least lwc
	q_min=lwc
	#
	# Maximum water vapor mixing ratio is unlimited, but should be smaller
	# than that for a cloud which base is at surface
	t_surf=theta*(p_surf/p00)**rcp
	q_max=calc_sat_mixr(p_surf,t_surf)
	#
	k=0
	while k<100:
		q_new=(q_min+q_max)/2
		lwc_calc=calc_lwc_altitude(p_surf,theta,q_new,zz)
		#	
		if abs(lwc-lwc_calc)<1e-7:
			break
		elif lwc<lwc_calc:
			q_max=q_new
		else:
			q_min=q_new
		k+=1
		# Failed
		if k==50: return -999
	#
	return q_new


def calc_lwp(p_surf,theta,pblh,rt):
	# Calculate liquid water path (kg/m^2) when boundary layer liquid water potential temperature (theta [K]) and total
	# water mixing ratio (rt [kg/kg]) are constants from surface (p_surf, Pa) up to boundary layer top (pblh, Pa or km).
	# In addition to the liquid water path, function returns cloud base and top heights (m) and the maximum (or cloud top)
	# liquid water mixing ratio (kg/kg).
	#
	# Constants
	R=287.04	# Specific gas constant for dry air (R_specific=R/M), J/kg/K
	Rm=461.5	# -||- for water
	ep2=Rm/R-1.0 #M_air/M_water-1
	cp=1005.0	# Specific heat for a constant pressure
	rcp=R/cp
	g=9.8
	p00=1.0e+05
	alvl = 2.5e+06 #  ! latent heat of vaporization
	#
	# It is assumed that a pblh value smaller than 10 is in kilometers and a value larger than that is Pa
	if pblh<10.0:
		z_top=pblh*1000. # from km to m (above surface)
		p_top=0.
	else:
		z_top=10e3
		p_top=p_surf-pblh # Pa (above surface)
	#
	# Outputs
	lwp=0.	# Liquid water path (g/m^2)
	zb=-999.	# Cloud base height (m)
	zc=-999.	# Cloud top height (m)
	clw_max=0.	# Maximum cloud liquid water
	#
	# a) Integrate to cloud base altitude
	dz=1.			# 1 m resolution
	z=0.				# The first altitude
	press=p_surf	# Start from surface
	RH=0
	while press>p_top and z<=z_top:
		# Temperature (K) 
		tavg=theta*(press/p00)**rcp
		#
		# Current RH (%)
		RH=calc_rh(rt,tavg,press)
		if RH>100:
			zb=z
			break
		#
		# From z to z+dz
		z+=dz
		# Virtual temperature: T_virtual=T*(1+ep2*rl)
		xsi=(1+ep2*rt)
		# Pressure (Pa)
		press-=g*dz*press/(R*tavg*xsi)
	#
	# No cloud or cloud water
	if RH<=100: return lwp,zb,zc,clw_max
	zb=z
	#
	# b) Integrate up to the given altitude
	while press>p_top and z<=z_top:
		# From z to z+dz
		z+=dz
		#
		# Moist adiabatic lapse rate
		#q_sat=calc_sat_mixr(press,tavg)
		q_sat=calc_sat_mixr(press,tavg)
		tavg-=g*(1+alvl*q_sat/(R*tavg))/(cp+alvl**2*q_sat/(Rm*tavg**2))*dz
		#
		# New pressure
		xsi=(1+ep2*q_sat)
		press-=g*dz*press/(R*tavg*xsi)
		#
		# Cloud water mixing ratio = totol - vapor
		rc=max(0.,rt-q_sat)
		# LWP integral
		lwp+=rc*dz*press/(R*tavg*xsi)
	#
	# Cloud top height
	zc=z
	clw_max=rc
	#
	# Return LWP (kg/m^2) and boundary layer height (m)
	return lwp,zb,zc,clw_max


def solve_rw_lwp(p_surf,theta,lwp,pblh,debug=False):
	# Solve boundary layer total water mixing ratio (kg/kg) from liquid water potential temperature (theta [K]),
	# liquid water path (lwp, kg/m^2) and boundary layer height (pblh, Pa or km) for an adiabatic cloud.
	# For example, solve_rw_lwp(101780.,293.,100e-3,20000.) would return 0.00723684088331 [kg/kg].
	#
	# Constants
	R=287.04	# Specific gas constant for dry air (R_specific=R/M), J/kg/K
	cp=1005.0	# Specific heat for a constant pressure
	rcp=R/cp
	p00=1.0e+05
	#
	# LWP tolerance: 0.1 % but not more than 0.1e-3 kg/m^2 and not less than 1e-3 kg/kg
	tol=min(max(0.001*lwp,0.1e-3),1e-3)
	#
	# Surface temperature (dry, i.e. no fog)
	t_surf=theta*(p_surf/p00)**rcp
	#
	# The highest LWP when RH=100% at the surface (no fog)
	rw_max= calc_sat_mixr(p_surf,t_surf)
	lwp_max,zb,zc,clw_max=calc_lwp(p_surf,theta,pblh,rw_max)
	# No fog cases
	if lwp_max<lwp:
		if debug: print ('Too high LWP (%5.1f g/m2), the maximum is %5.1f g/m2 (theta=%6.2f K, pblh=%3.0f hPa)')%(lwp*1e3, lwp_max*1e3,theta,pblh/100.)
		return -999.
	#
	# The lowest LWP when RH=0% at the surface
	rw_min=0.
	lwp_min,zb,zc,clw_max=calc_lwp(p_surf,theta,pblh,rw_min)
	if lwp_min>lwp:
		if debug: print ('Too low LWP (%5.1f g/m2), the minimum is %5.1f g/m2 (theta=%6.2f K, pblh=%3.0f hPa)')%(lwp*1e3, lwp_max*1e3,theta,pblh/100.)
		return -999.
	#
	k=0
	while k<100:
		rw_new=(rw_min+rw_max)*0.5
		lwp_new,zb,zc,clw_max=calc_lwp(p_surf,theta,pblh,rw_new)
		#
		if abs(lwp-lwp_new)<tol or abs(rw_max-rw_min)<0.001e-3:
			return rw_new
		elif lwp<lwp_new:
			rw_max=rw_new
		else:
			rw_min=rw_new
		k+=1
	#
	# Failed
	if debug: print ('Iteration failed: current LWP=%5.1f, target LWP=%5.1f')%(lwp_new*1e3,lwp*1e3)
	return -999.


def solve_q_inv_RH(press,tpot,q,max_RH):
	# Function for adjusting total water mixing ratio so that the calculated RH will be no more
	# than the given RH limit. This function can be used to increase humidity inversion so that RH
	# above cloud is less than 100%. For this purpose the typical inputs are:
	#	press [Pa] = p_surf - pblh
	#	tpot [K] = tpot_pbl + tpot_inv
	#	q [kg/kg] = q_pbl - q_inv
	#	RH [%] = 98.
	#
	# Constants
	R=287.04	# Specific gas constant for dry air (R_specific=R/M), J/kg/K
	cp=1005.0	# Specific heat for a constant pressure
	rcp=R/cp
	p00=1.0e+05
	#
	# Temperature (K)
	temp=tpot*(press/p00)**rcp
	#
	# RH (%)
	rh=calc_rh(q,temp,press)
	#
	# All done if RH is not exceeding the RH limit
	if rh<=max_RH: return q, rh, rh
	#
	# Solve q so that RH=max_RH
	q_min=0.
	q_max=q
	k=0
	while k<200:
		q_new=0.5*(q_min+q_max)
		rh_new=calc_rh(q_new,temp,press)
		#
		if abs(rh_new-max_RH)<0.001:
			return q_new, rh_new, rh
		elif rh_new>max_RH:
			q_max=q_new
		else:
			q_min=q_new
		k+=1
	#
	# Failed
	print ('Failed to solve water vapor mixing ratio from given RH!')
	return -999.,-999., rh


#
#
#
# ================ Helper functions ================
#
def ls_fit(xx,yy):
	# Simple linear least squares fit: y=a+b*x
	import numpy
	#
	# Ignore NaN's
	x=[]; y=[]
	i=0
	for val in xx:
		if not (numpy.isnan(xx[i]) or numpy.isnan(yy[i])):
			x.append(xx[i])
			y.append(yy[i])
		i+=1
	#
	if len(x)<=1:
		# Scalar
		a=0.0; a_std=0.0
		b=1.0; b_std=0.0
	else:
		# Matrix H
		H = numpy.matrix( numpy.vstack([numpy.ones(len(x)),x]).T )
		# LS solution
		th=numpy.linalg.inv( H.T*H ) * H.T *numpy.matrix(y).T
		# Outputs
		a=numpy.asscalar(th[0])
		b=numpy.asscalar(th[1])
		# Parameter uncertainty
		if len(x)>2:
			# Variance
			sv2=((numpy.matrix(y).T-H*th).T * (numpy.matrix(y).T-H*th) )/(len(x)-2)
			std=numpy.sqrt( numpy.asscalar(sv2) * numpy.diagonal( numpy.linalg.inv( H.T*H ) ) )
			# Outputs
			a_std=numpy.asscalar(std[0])
			b_std=numpy.asscalar(std[1])
		else:
			a_std=0.0
			b_std=0.0
	# 
	return a,b,a_std,b_std,

def average_scaled(x,y):
	# Calculate average of x/y so that points where y=0 are ignored
	import numpy
	sx=0.
	sx2=0.
	n=0
	i=0
	for yy in y:
		if yy>0.:
			sx+=x[i]/yy
			sx2+=(x[i]/yy)**2
			n+=1
		i+=1
	#
	if n==0:
		return -999., -999.
	elif n==1:
		return sx, -999.
	else:
		return sx/n, numpy.sqrt( sx2/n - (sx/n)**2 )

#
# Functions from the LES model
# 

def calc_psat_w(T):
	# Function calculates the saturation vapor pressure (Pa) of liquid water as a function of temperature (K)
	#
	# thrm.f90:  real function rslf(p,t)
	c0=0.6105851e+03
	c1=0.4440316e+02
	c2=0.1430341e+01
	c3=0.2641412e-01
	c4=0.2995057e-03
	c5=0.2031998e-05
	c6=0.6936113e-08
	c7=0.2564861e-11
	c8=-.3704404e-13
	#
	x=max(-80.,T-273.16)
	return c0+x*(c1+x*(c2+x*(c3+x*(c4+x*(c5+x*(c6+x*(c7+x*c8)))))))


def calc_sat_mixr(p,T):
	# Function calculates saturation mixing ratio for water (kg/kg)
	#
	# thrm.f90: real function rslf(p,t)
	#
	# r=m_w//m_air
	# R/Rm=287.04/461.5=.622
	#
	esl=calc_psat_w(T)
	return .622*esl/(p-esl)


def calc_rh(rw,T,press):
	# Calculate RH (%) from water vapor mixing ratio rw (r=m_w/m_air [kg/kg]), temperature (K) and pressure (Pa)
	#
	# r=m_w//m_air=pw/Rm/(pair/R)=pw/(p-pw)*R/Rm => pw=p*r/(R/Rm+r)
	#
	R=287.04	# Specific gas constant for dry air (R_specific=R/M), J/kg/K
	Rm=461.5	# Specific gas constant for water
	ep=R/Rm
	#
	psat=calc_psat_w(T)
	return press*rw/(ep+rw)/psat*100
	# When ep>>rw => RH=press*rw/(ep*psat)*100
# ================================