import h5py
import pprint

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy

from functools import wraps
from time import time

import scipy
import scipy.signal
import scipy.spatial

def import_mono(path_mono, hkls_equiv):

    with h5py.File(path_mono, 'r') as mono:
        h_mono = np.array(mono['mono']['h_unit'])
        intensities = np.array(mono['mono']['F_squared_meas'])

    vecs_cart = h_mono
    for equiv in hkls_equiv:
        new_hkls = h_mono * equiv
        vecs_cart = np.concatenate((vecs_cart, new_hkls))

    intensites_full = intensities
    for i in range (len(hkls_equiv)):
        intensites_full = np.append(intensites_full, intensities)

    vecs_sph = []
    for vec_cart in vecs_cart:
        vec_sph = cartesian_to_spherical(*vec_cart)
        vecs_sph.append(vec_sph[1:])
    vecs_sph = np.array(vecs_sph)

    return vecs_cart, vecs_sph, intensites_full

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return [r, theta, phi]

def arr_cartesian_to_spherical(input_array):
    vecs_sph = []
    for vec_cart in input_array:
        vec_sph = cartesian_to_spherical(*vec_cart)
        vecs_sph.append(vec_sph[1:])
    vecs_sph = np.array(vecs_sph)
    return vecs_sph

def spherical_to_cartesian(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return [x, y, z]

def arr_spherical_to_cartesian(input_array):
    vecs_cart = []
    for vec_sph in input_array:
        vec_cart = spherical_to_cartesian(*vec_sph)
        vecs_cart.append(vec_cart)
    vecs_cart = np.array(vecs_cart)
    return vecs_cart

def draw_sphere(data_cart, output_png):

    vecs_cart = data_cart
    U, V, W = zip(*vecs_cart)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(U, V, W, s=2, marker = '.', lw = 0)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plt.savefig(output_png, dpi = 1200)

def rotate_sphere(data_cart, rot_matrix):
    vecs_rotated = []
    for vec_cart in data_cart:
        vec_rotated = np.matmul(rot_matrix, vec_cart)
        vecs_rotated.append(vec_rotated)
    vecs_rotated = np.array(vecs_rotated)
    return vecs_rotated

path_expt = r'C:\Users\piotr\Documents\VS_Code\omatrix\testdata\dark_01__expt.h5'
path_mono = r'C:\Users\piotr\Documents\VS_Code\omatrix\testdata\Rh1_mono.h5'
path_xxx = r'C:\Users\piotr\Documents\VS_Code\omatrix\testdata\xxx.h5'
hkls_equiv = ((-1, -1, -1), (-1, 1, -1), (1, -1, 1))

rotation_R = np.array([[1, 0, 0],
                      [0, 0, -1],
                      [0, 1, 0]])

org_mono_cart, org_mono_sph, intensities = import_mono(path_mono, hkls_equiv)
shift_mono_cart = rotate_sphere(org_mono_cart, rotation_R)

corr = scipy.signal.correlate(org_mono_cart, shift_mono_cart)

y, x, z = np.unravel_index(np.argmax(corr), corr.shape) 

print(x,y,z)
pass


# draw_sphere(org_mono_cart, 'org.png')
# draw_sphere(shift_mono_cart, 'shift.png')

##THRESHOLDING
# rays = np.concatenate((org_mono_cart, intensities[:, np.newaxis]), axis = 1)

# threshold_min = 20000
# threshold_max = 100000
# rays_filtered = np.array([ray for ray in rays if ray[3] > threshold_min and ray[3] < threshold_max])

# print (len(rays_filtered))

# rays_filtered_cart = rays_filtered[:,:3]

# draw_sphere(rays_filtered_cart, 'threshed.png')