"""
ADAPTED FROM ENLIB (SIGURD NAESS)

`python pksc2hdf5.py {input.pksc} {output.h5} {num_halos}`

i.e. for a test case with only ONE HUNDRED HALOS
```
python pksc2hdf5.py /fs/lustre/project/act/mocks/websky/v0.0/halos.pksc only_a_hundred.h5 100
```

Omit the number of halos to convert *all of them*.
```
python pksc2hdf5.py /fs/lustre/project/act/mocks/websky/v0.0/halos.pksc websky_all.h5
```
"""

import sys
import numpy as np
import pyccl
import h5py
from pixell import utils, bunch, enmap

def websky_pkcs_read(fname, num=0, offset=0):
	"""Read rows offset:offset+num of raw data from the given pkcs file.
	if num==0, all values are read"""
	with open(fname, "r") as ifile:
		n   = np.fromfile(ifile, count=3, dtype=np.uint32)[0]-offset
		if num: n = num
		cat = np.fromfile(ifile, count=n*10, offset=offset*10*4, dtype=np.float32).reshape(n, 10)
		return cat

def websky_decode(data, cosmology, mass_interp):
	"""Go from a raw websky catalog to pos, z and m200"""
	chi     = np.sum(data.T[:3]**2,0)**0.5 # comoving Mpc
	a       = pyccl.scale_factor_of_chi(cosmology, chi)
	z       = 1/a-1
	R       = data.T[6].astype(float) * 1e6*utils.pc # m. This is *not* r200!
	rho_m   = calc_rho_c(0, cosmology)*cosmology["Omega_m"]
	m200m   = 4/3*np.pi*rho_m*R**3
	m200c    = mass_interp(m200m, z)
	ra, dec = utils.rect2ang(data.T[:3])
	return bunch.Bunch(z=z, ra=ra, dec=dec, m200m=m200m, m200c=m200c, pos=data.T[:3])


def get_H0(cosmology): return cosmology["h"]*100*1e3/(1e6*utils.pc)

def get_H(z, cosmology):
	z = np.asanyarray(z)
	return get_H0(cosmology)*pyccl.h_over_h0(cosmology, 1/(z.reshape(-1)+1)).reshape(z.shape)

def calc_rho_c(z, cosmology):
	H     = get_H(z, cosmology)
	rho_c = 3*H**2/(8*np.pi*utils.G)
	return rho_c

class MdeltaTranslator:
	def __init__(self, cosmology,
			type1="matter", delta1=200, type2="critical", delta2=200,
			zlim=[0,20], mlim=[1e11*utils.M_sun,5e16*utils.M_sun], step=0.1):
		"""Construct a functor that translates from one M_delta defintion to
		another.
		* type1, type2: Type of M_delta, e.g. m200c vs m200m.
		  * "matter": The mass inside the region where the average density is
		    delta times higher than the current matter density
		  * "critical": The same, but for the critical density instead. This
		    differs due to the presence of dark energy.
		* delta1, delta2: The delta value used in type1, type2.
		* zlim: The z-range to build the interpolator for.
		* mlim: The Mass range to build the interpolator for, in kg
		* step: The log-spacing of the interpolators.
		Some combinations of delta and type may not be supported, limited by
		support in pyccl. The main thing this object does beyond pyccl is to
		allow one to vectorize over both z and mass."""
		idef = pyccl.halos.MassDef(delta1, type1, c_m_relation="Bhattacharya13")
		odef = pyccl.halos.MassDef(delta2, type2, c_m_relation="Bhattacharya13")
		# Set up our sample grid, which will be log-spaced in both z and mass direction
		lz1, lz2 = np.log(1+np.array(zlim)) # lz = log(1+z) = -log(a)
		lm1, lm2 = np.log(np.array(mlim))   # lm = log(m)
		nz  = utils.ceil((lz2-lz1)/step)
		nm  = utils.ceil((lm2-lm1)/step)
		lzs = np.linspace(lz1, lz2, nz)
		lms = np.linspace(lm1, lm2, nm)
		olms = np.zeros((len(lzs),len(lms)))
		for ai, lz in enumerate(lzs):
			moo = np.exp(lms[-1])/utils.M_sun
			olms[ai] = idef.translate_mass(cosmology, np.exp(lms)/utils.M_sun, np.exp(-lz), odef)
		olms = np.log(olms*utils.M_sun)
		olms = utils.interpol_prefilter(olms, order=3)
		# Save parameters
		self.lz1, self.lz2, self.dlz = lz1, lz2, (lz2-lz1)/(nz-1)
		self.lm1, self.lm2, self.dlm = lm1, lm2, (lm2-lm1)/(nm-1)
		self.olms = olms
	def __call__(self, m, z):
		zpix = (np.log(1+np.array(z))-self.lz1)/self.dlz
		mpix = (np.log(m)-self.lm1)/self.dlm
		pix  = np.array([zpix,mpix])
		return np.exp(utils.interpol(self.olms, pix, order=3, prefilter=False))


argshalos = sys.argv[1]
outfile = sys.argv[2]

nhalos = 0
if len(sys.argv) == 4:  # 0=script, 1=infile, 2=outfile, 3=nhalos
    nhalos = int(sys.argv[3])

dtype = np.float32
box = np.array([[-5,10],[5,-10]]) * utils.degree
shape,wcs = enmap.geometry(pos=box,res=0.5 * utils.arcmin,proj='car')
omap        = enmap.zeros(shape[-2:], wcs, dtype)

cosmology   = pyccl.Cosmology(Omega_c=0.2589, Omega_b=0.0486, h=0.6774, sigma8=0.8159, n_s=0.9667, transfer_function="boltzmann_camb")
mass_interp = MdeltaTranslator(cosmology)

data  = websky_pkcs_read(argshalos, nhalos)
cat    = websky_decode(data, cosmology, mass_interp); del data

f = h5py.File(outfile, 'a')
f["ra"] = cat.ra[:,None].astype(np.float32)
f["dec"] = cat.dec[:,None].astype(np.float32)
f["m200c"] = (cat.m200c[:,None] / utils.M_sun).astype(np.float32)
f["m200m"] = (cat.m200m[:,None] / utils.M_sun).astype(np.float32)
f["z"] = cat.z[:,None].astype(np.float32)
f["pos"] = cat.pos
f.close()
