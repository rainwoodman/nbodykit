import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.optimize import minimize
from scipy.sparse import csr
from astropy.cosmology import Planck15
from astropy.units import Mpc, km, s

from nbodykit import CurrentMPIComm
from nbodykit.dataset import DataSet
from nbodykit.meshtools import SlabIterator
from pmesh.pm import ComplexField
from nbodykit.source.mesh import LinearMesh
from nbodykit.source.mesh import BigFileMesh
from nbodykit.io.bigfile import BigFile



class FGPA(object):
    logger = logging.getLogger('FGPA')
    
    def __init__(self, source, z, seed, Nmesh, BoxSize):
        # setting parameters
        self.attrs = {}
        self.attrs['z'] = z
        self.attrs['seed'] = seed
        self.attrs['Nmesh'] = Nmesh  
        self.attrs['BoxSize'] = BoxSize

        # import data from simulation
        self.particles = source
        self.mesh = self.particles.to_mesh(Nmesh=Nmesh, BoxSize=BoxSize)
        self.real = self.mesh.to_field(mode='real')


    # calculate overdensity field  
    def to_overdensity(self):
        self.delta_mean = np.mean(self.real)
        self.delta = self.real / self.delta_mean - 1
        return self.delta
    
    # calculate velocity field
    def get_cell_velocity(self):
        self.particles['vz'] = self.particles['Velocity'][:, 2]
        self.vreal_accum = self.particles.to_mesh(Nmesh=self.attrs['Nmesh'], BoxSize=self.attrs['BoxSize'] , weight='vz').to_field(mode='real')
        self.velocity = np.where(self.real[:] != 0., self.vreal_accum[:]/self.real[:], 0)
        return self.velocity

    # apply Gaussian smoothing to real field to get baryons instead of DM
    # smoothing = 0.07: random parameter that makes sense
    # (100 kpc in bi and davidsen http://iopscience.iop.org/article/10.1086/303908/pdf)
    def smoothing(self, smoothing):
        self.attrs['smoothing'] = smoothing
        self.real = self.real.r2c().apply(lambda k, v: v * np.exp(- 0.5 * self.attrs['smoothing'] ** 2 * sum(ki **2 for ki in k))).c2r()    
        return self.real

    # temperature-density-relation to calculate T field
    def temperature_field(self, T_IGM, gamma):
        self.attrs['T_IGM'] = T_IGM
        self.attrs['gamma'] = gamma
        self.T = self.attrs['T_IGM'] * ( 1 + self.delta ) ** (self.attrs['gamma'] - 1.)
        return self.T
    
    def set_A(self, A):
        self.attrs['A'] = A
        return self.attrs['A']
    
def sample_PS(fgpa_obj, which_field, number_of_skewers, random_seed):
    F_mean_fiducial = flux_fiducial(fgpa_obj.attrs['z'])[0]
    los = create_los(fgpa_obj, which_field, number_of_skewers, randon_seed, F_mean_fiduacial)
    PS_1d = power_spectrum_1d(los, fgpa_obj.attrs['z'], fgpa_obj.attrs['BoxSize'], fgpa_obj.attrs['Nmesh'])
    return PS_1d

# calculate fiducial values for flux and variance of flux
def flux_fiducial(z):
    F_mean_fiducial = np.exp(-0.001845*(1 + z)**3.924)
    F_var_fiducial = 0.065*((1 + z)/3.25)**3.8 * F_mean_fiducial**2
    return F_mean_fiducial, F_var_fiducial

# calculate 1d power spectrum
def power_spectrum_1d(fgpa_obj, los):
    
    los_fft = np.fft.rfft(los, axis = 1) / fgpa_obj.attrs['Nmesh']
    los_fft_abs_sqr = np.absolute(los_fft) ** 2 
    
    vel_to_dist = (fgpa_obj.attrs['z'] + 1) / Planck15.H(fgpa_obj.attrs['z']) / (Mpc/Planck15.h) * km / s

    PS_1d = np.mean(los_fft_abs_sqr, axis = 0) * (fgpa_obj.attrs['BoxSize']/vel_to_dist) 
    
    return PS_1d


def create_los(fgpa_obj, which_field, number_of_skewers, random_seed):
    np.random.seed(random_seed)
    xy_skewer_pos_array = np.int32(np.random.uniform(size = (number_of_skewers,2)) * fgpa_obj.attrs['Nmesh'])
    
    los = np.zeros((len(xy_skewer_pos_array), fgpa_obj.attrs['Nmesh']))
    for i in range(len(xy_skewer_pos_array)):
        los[i] = line_of_sight(fgpa_obj, xy_skewer_pos_array[i])[which_field]
    return los


# calculate optical depth and flux (with and without RSD and thermal broadening) along a skewer in z direction
def line_of_sight(fgpa_obj, xy_skewer_pos):
    
    # pick a skewer out of the simulated 3D box
    xy_skewer_pos = tuple(xy_skewer_pos)
    delta_z = fgpa_obj.delta[xy_skewer_pos]
    velocity_z = fgpa_obj.velocity[xy_skewer_pos]
    T_z = fgpa_obj.T[xy_skewer_pos]
    
    vel_to_dist = (fgpa_obj.attrs['z'] + 1) / Planck15.H(fgpa_obj.attrs['z']) / (Mpc/Planck15.h) * km / s
    SQRT_KELVIN_TO_KMS = 0.12849 #sqrt(2 * k_B / m_proton) in km/s
    
    # calculate RSD
    dreal = np.linspace(0, fgpa_obj.attrs['BoxSize'] - fgpa_obj.attrs['BoxSize'] / fgpa_obj.attrs['Nmesh'], fgpa_obj.attrs['Nmesh'])
    dRSD = velocity_z * vel_to_dist
    dred = dreal + dRSD

    # calculate thermal broadening
    vtherm = SQRT_KELVIN_TO_KMS * np.sqrt(T_z)
    dtherm = vtherm * vel_to_dist
    
    # calculate optical depth tau_real from density with fluctuating Gunn-Peterson approximation
    tau_real_z = fgpa(delta_z, fgpa_obj.attrs['A'], fgpa_obj.attrs['gamma'])
    
    # calculate optical depth with RSD and thermal broadening using funky convolution with Gaussians
    tau_red_z = irconvolve(dreal, dred, tau_real_z, dtherm)
    
    # calculate flux from optical depth
    F_real_z = np.exp(- tau_real_z)
    F_red_z = np.exp(- tau_red_z)
    F_mean_fiducial = flux_fiducial(fgpa_obj.attrs['z'])[0]
    F_red_normalized_z = F_red_z / F_mean_fiducial - 1

    return {'delta':delta_z,
            'tau_real':tau_real_z,
            'tau_red':tau_red_z,
            'F_real':F_real_z,
            'F_red':F_red_z,
            'F_red_normalized':F_red_normalized_z}


# Fluctuating Gunn-Peterson approximation to obtain optical depth from density field
def fgpa(delta, A, gamma):
    tau = A * (1 + delta) ** ( 2 - 0.7 * (gamma - 1) )
    return tau
    
    
# Yu's fancy convolution function
def irconvolve(xc, x, y, h, 
        kernel=lambda r, h: np.exp(- 0.5 * (r / h) ** 2)):
    """ default kernel is gaussian
        exp - 1/2 * r / h
        xc has to be uniform!
    """
    xc, y, x, h = np.atleast_1d(xc, y, x, h)
    dxc = (xc[-1] - xc[0]) / (len(xc) - 1)
    support = 6

    #first remove those are too far off
    good = ((x + support * h > xc[0]) \
          & (x - support * h < xc[-1]))
    x = x[good]
    y = y[good]
    h = h[good]

    if len(h) > 0:
        # the real buffer is bigger than out to ease the normalization
        # still on the edge we are imperfect
        padding = int((2 * support + 1)* h.max() / dxc) + 1
        padding = max(padding, 2)
        buffer = np.zeros(shape=len(xc) + 2 * padding)
        paddedxc = np.empty(buffer.shape, dtype=xc.dtype)
        paddedxc[padding:-padding] = xc
        # here comes the requirement xc has to be uniform.
        paddedxc[:padding] = xc[0] - np.arange(padding, 0, -1) * dxc
        paddedxc[-padding:] = xc[-1] + np.arange(1, padding +1) * dxc
        out = buffer[padding:-padding]
        assert len(out) == len(xc)
        assert (paddedxc[1:] > paddedxc[:-1]).all()

        # slow. for uniform xc/paddedxc, we can do this faster than search
        start = paddedxc.searchsorted(x - support * h, side='left')
        end = paddedxc.searchsorted(x + support * h, side='left')
        # tricky part, build the csr matrix for the conv operator,
        # only for the non-zero elements (block diagonal)
        N = end - start + 1
        indptr = np.concatenate(([0], N.cumsum()))
        indices = np.repeat(start - indptr[:-1], N) + np.arange(N.sum())
        r = np.repeat(x, N) - paddedxc[indices]
        data = kernel(r, np.repeat(h, N))
        data[np.repeat(N==1, N)] = 1
        data[np.repeat(h==0, N)] = 1
        matrix = csr.csr_matrix((data, indices, indptr), 
                shape=(len(x), len(paddedxc)))
        norm = np.repeat(matrix.sum(axis=1).flat, N)
        data /= norm
        buffer[:] = matrix.transpose() * y
    else:
        out = np.zeros(shape=xc.shape, dtype=y.dtype)
    return out


def fitting_method(fgpa_obj, A_initial, gamma_initial, which_field, number_of_skewers, random_seed, T_IGM):

    def model(A, gamma, fgpa_obj, which_field, number_of_skewers, random_seed, T_IGM):
        fgpa_obj.set_A(A)
        fgpa_obj.temperature_field(fgpa_obj, T_IGM, gamma)
        F = create_los(fgpa_obj, which_field, number_of_skewers, random_seed)
        F_mean = np.mean(F)
        F_var = np.var(F)
        return F_mean, F_var

    def objective(x, which_field, number_of_skewers, random_seed, fgpa_obj, T_IGM):
        F_mean, F_var = model(x[0], x[1], fgpa_obj, which_field, number_of_skewers, random_seed, T_IGM)
        F_mean_fiducial, F_var_fiducial = flux_fiducial(fgpa_obj.attrs['z'])
        chi_squared_F_mean = (F_mean_fiducial - F_mean)**2/F_mean_fiducial**2
        chi_squared_F_var = (F_var_fiducial - F_var)**2/F_var_fiducial**2
        return chi_squared_F_mean  + chi_squared_F_var

    res = minimize(objective, (A_initial,gamma_initial),
                   args = (which_field, number_of_skewers, random_seed, T_IGM),
                   method = 'Nelder-Mead')
    A, gamma = res.x[0], res.x[1]
    
    return A, gamma
    

def fit(simulation_file, boxsize, number_of_pixels, z, smoothing, T_IGM,
                    A_initial, gamma_initial, which_field, number_of_skewers, random_seed):
    
    F_mean_fiducial, F_var_fiducial = flux_fiducial(z)

    particle_pos, particle_vel = import_data(simulation_file)
    delta, velocity, T_junk = painting(particle_pos = particle_pos, particle_vel = particle_vel,
                                  boxsize = boxsize, number_of_pixels = number_of_pixels,
                                  smoothing = smoothing, T_IGM = T_IGM, gamma = gamma_initial)
    A, gamma = fit(A_initial, gamma_initial, which_field, number_of_skewers, random_seed,
                   delta, velocity, z, boxsize, number_of_pixels, F_mean_fiducial, F_var_fiducial, T_IGM)
    return A, gamma

