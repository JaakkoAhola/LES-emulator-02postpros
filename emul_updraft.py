# Note 1: Voima requires python 2.7.14
#  module load Python/2.7.14
#
# Note 2: Level 4 radiative flux and Level 3 CDNC are not 3D outputs!
#


import numpy

#
# ************************* Functions *************************
# From LES2emu.py
#

def get_netcdf_updraft(fname,tstart,tend,ttol=3600.,tol_clw=1e-5):
    # Function for calculating mean positive updraft velocities and cloud droplet number concentrations
    # at cloud base (see Romakkaniemi et al., 2009) from 4D data (*.nc).
    #
    # Romakkaniemi, S., G. McFiggans, K.N. Bower, P. Brown, H. Coe, T.W. Choularton, A comparison between
    # trajectory ensemble and adiabatic parcel modelled cloud properties and evaluation against airborne
    # measurements, J. Geophys. Res., doi:10.1029/2008JD011286, 2009
    #
    # Inputs:
    #  fname           Complete file path and name (*.nc)
    #  start, tend     Averaging time window (s)
    # Optional inputs
    #  ttol            Time tolelance (s) for finding averaging window
    #  tol_clw         Cloud liquid water mixing ratio (kg/kg) for the cloud base
    #
    import os
    import netCDF4 as netcdf
    import numpy
    #
    # File must exist
    if not os.path.lexists(fname): raise RuntimeError(fname+' not found!')
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
    # Optional: CDNC and radiative flux
    cdnc_calc = ('S_Nc' in ncid.variables)
    rflx_calc = ('rflx' in ncid.variables)
    #
    #
    # Outputs
    wpos=0.    # Mean positive updraft velocity at the cloud base (m/s)
    w2pos=0.   # Velocity weighted mean positive updraft velocity at the cloud base (m/s)
    cdnc_p=0.  # Mean cloud droplet number concentration at the cloud base with positive updraft velocity (1/kg)
    cdnc_wp=0. # Velocity weigted mean cloud droplet number concentration at the cloud base with positive updraft velocity (1/kg)
    n=0        # Number of cloud bases with positive updraft (-)
    drflx=0.   # Cloud radiative forcing (W/m^2); for all cloudy columns
    m=0        # Number of cloudy colums (-)
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
        if cdnc_calc:
            if ('S_Nc' in ncid.variables):
                cdnc=ncid.variables['S_Nc'][ind,]
            else:
                cdnc=ncid.variables['CCN'][ind,]
        if rflx_calc:
            rflx=ncid.variables['rflx'][ind,]
        #
        # Calculations
        for i in range(0,dims[0]):
            for j in range(0,dims[1]):
                kk, = numpy.where(l[i,j,:]>tol_clw)
                if len(kk)>0 and w[i,j,kk[0]]>0.:
                    # Cloud and positive updraft velocity
                    k=kk[0]
                    n+=1
                    wpos+=w[i,j,k]
                    w2pos+=w[i,j,k]**2
                    if cdnc_calc:
                        cdnc_p+=cdnc[i,j,k]
                        cdnc_wp+=w[i,j,k]*cdnc[i,j,k]
                if len(kk)>0 and rflx_calc:
                    # Cloud; radiative cooling effect
                    m+=1
                    k=kk[0]; h=kk[-1] # The first and last cloudy grid cell
                    drflx+=rflx[i,j,k]-rflx[i,j,h] # Radiative warming: in - out
    #
    if n>0:
        w2pos/=wpos
        wpos/=n
        if cdnc_calc:
            cdnc_p/=n
            cdnc_wp/=(wpos*n)
    if rflx_calc and m>0:
        drflx=drflx/m
    #
    # Close file
    ncid.close()
    #
    # Outputs: mean positive updraft velocity and cloud droplet number concentrations (mean and
    # weighted with velocity) at the cloud base, and mean cloud radiative effect
    return wpos,w2pos,cdnc_p,cdnc_wp,n,drflx,m




#
# ************************* Calculations *************************
#

# Input data
opt = 4
if opt==1:
    # Level 3
    # BSP design (3.0.0) + level 3 + night (Jan 2019)
    path='/arch/eclair/UCLALES-SALSA_training_simulations/case_emulator_DESIGN_v3.0.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_night'
    n=500
    out_name='updraft_L3_N.txt'
elif opt==2:
    # BSP design (3.1.0) + level 3 + day (Jan 2019)
    path='/arch/eclair/UCLALES-SALSA_training_simulations/case_emulator_DESIGN_v3.1.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_day'
    n=500
    out_name='updraft_L3_D.txt'
elif opt==3:
    # Level 4
    # BSP design (3,2,3 & 3.2.4) + level 4 + night (9.4.2019 & 6.6.2019)
    path='/arch/eclair/UCLALES-SALSA_training_simulations/case_emulator_DESIGN_v3.2_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_night'
    n=135
    out_name='updraft_L4_N.txt'
else:
    # BSP design (3,3,2 & 3.3.3) + level 4 + day (9.4.2019 & 6.6.2019)
    path='/arch/eclair/UCLALES-SALSA_training_simulations/case_emulator_DESIGN_v3.3_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_day'
    n=150
    out_name='updraft_L4_D.txt'
fmt='/emul%03u/emul%03u'
#
# Time window
tstart=2.5*3600
tend=3.5*3600
#
# Outputs
out=[]
#
for i in range(1,n+1):
    # Data file
    file=(path+fmt+'.nc') % (i,i)
    #
    wpos,w2pos,cdnc_p,cdnc_wp,n,drflx,m = get_netcdf_updraft(file,tstart,tend,ttol=10.)
    #
    print(i,wpos,w2pos,cdnc_p,cdnc_wp,n,drflx,m)
    #
    if '_LVL4' in path:
        # No 3d radiative flux data for Level 4
        out.append([wpos,w2pos,cdnc_p,cdnc_wp,n])
    else:
        # No 3d CDNC data for Level 3
        out.append([wpos,w2pos,n,drflx,m])
#
numpy.savetxt(out_name,out,delimiter=' ',newline='\n')
