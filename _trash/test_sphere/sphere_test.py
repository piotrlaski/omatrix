import h5py
import pprint

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy

from functools import wraps
from time import time

import scipy
import scipy.spatial

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap


path_expt = r'C:\Users\piotr\Documents\VS_Code\omatrix\testdata\dark_01__expt.h5'
path_mono = r'C:\Users\piotr\Documents\VS_Code\omatrix\testdata\Rh1_mono.h5'
path_xxx = r'C:\Users\piotr\Documents\VS_Code\omatrix\testdata\xxx.h5'
hkls_equiv = ((-1, -1, -1), (-1, 1, -1), (1, -1, 1))

def spherical_distance(u, v):

    theta1, phi1 = u
    theta2, phi2 = v

    cos_d = (np.sin(theta1) * np.sin(theta2) + np.cos(theta1) * np.cos(theta2) * np.cos(phi1 - phi2))
    cos_d = np.clip(cos_d, -1, 1)
    d = np.arccos(cos_d)
    return d

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return [r, theta, phi]

def spherical_to_cartesian(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return [x, y, z]

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

    ax.view_init(0, 120)
    plt.savefig(output_png, dpi = 1200)

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

def import_expt(path_expt):

    with h5py.File(path_expt, 'r') as expt:
        h_expt = np.array(expt['expt']['h_unit_g'])   
        intensities = np.array(expt['expt']['I'])
    vecs_cart = h_expt

    vecs_sph = []
    for vec_cart in vecs_cart:
        vec_sph = cartesian_to_spherical(*vec_cart)
        vecs_sph.append(vec_sph[1:])
    vecs_sph = np.array(vecs_sph)

    return vecs_cart, vecs_sph, intensities

def convert_distance_to_spherical(dist):
    return 2*np.arcsin(dist/2)

@timing
def cluster_hier(cluster_crit, cluster_nb_crit):
    vecs_cart_expt, vecs_sph_expt, intensities = import_expt(path_expt) 
    dist_mx_cart = scipy.spatial.distance.pdist(vecs_cart_expt)
    dist_mx = convert_distance_to_spherical(dist_mx_cart)
    Z = scipy.cluster.hierarchy.complete(dist_mx)
    clustered = scipy.cluster.hierarchy.fcluster(Z, cluster_crit, criterion='distance')
    cluster_nb = max(clustered)
    print (cluster_crit, cluster_nb)

    cluster_centers = []
    for i in range(1, cluster_nb + 1):
        cluster_points = vecs_sph_expt[clustered == i]
        if len(cluster_points) < cluster_nb_crit:
            continue
        theta_mean = np.mean(cluster_points[:,0])
        phi_mean = np.mean(cluster_points[:,1])
#        z_mean = np.mean(cluster_points[:,2])
        cluster_carts = spherical_to_cartesian(theta_mean, phi_mean)
        cluster_centers.append(cluster_carts)
    cluster_centers = np.array(cluster_centers)

    print (len(cluster_centers))
    draw_sphere(cluster_centers, f'clusters{cluster_crit}.png')
    return cluster_centers



clusters_expt_cart = cluster_hier(cluster_crit = 0.015, cluster_nb_crit = 3)





