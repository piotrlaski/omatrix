import h5py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R


from functools import wraps
from time import time

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
def convert_distance_to_spherical(dist):
    dist = np.clip(dist, 0, 2)
    return 2*np.arcsin(dist/2)
def cart_dist(v,u):
    return np.sqrt(np.sum((v - u) ** 2))


def draw_sphere(data_cart, output_png):

    if np.shape(data_cart)[1] == 4:
        vecs_cart = data_cart[:,:3]
    else:
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
    return
def rotate_sphere(data_cart, rot_matrix):

    vecs_rotated = []
    for vec_cart in data_cart:
        vec_rotated = np.matmul(rot_matrix, vec_cart)
        vecs_rotated.append(vec_rotated)
    vecs_rotated = np.array(vecs_rotated)
    return vecs_rotated

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

def cluster_hier(rays_cart, rays_sph, min_intensity, max_intensity, cluster_crit, cluster_nb_crit):
     
    data_cart = rays_cart[:,:3]
    data_sph = rays_sph[:,:2]

    dist_mx_cart = scipy.spatial.distance.pdist(data_cart)
    dist_mx = convert_distance_to_spherical(dist_mx_cart)
    Z = scipy.cluster.hierarchy.complete(dist_mx)
    clustered = scipy.cluster.hierarchy.fcluster(Z, cluster_crit, criterion='distance')
    cluster_nb = max(clustered)
    print (f'{cluster_nb} expt clusters were found with clustering distance criterion of d < {cluster_crit} rad (complete hierarchy linkage)')

    cluster_centers = []
    for i in range(1, cluster_nb + 1):
        cluster_points = rays_sph[clustered == i]
        cluster_points = cluster_points[cluster_points[:,2] > min_intensity]
        cluster_points = cluster_points[cluster_points[:,2] < max_intensity]
        if len(cluster_points) < cluster_nb_crit:
            continue
        theta_mean = np.mean(cluster_points[:,0])
        phi_mean = np.mean(cluster_points[:,1])
        cluster_carts = spherical_to_cartesian(theta_mean, phi_mean)
        cluster_centers.append(cluster_carts)
    cluster_centers = np.array(cluster_centers)
    print (f'{len(cluster_centers)} expt clusters are left after filtering (at least {cluster_nb_crit} vecs/cluster, all satisfying: {min_intensity} < PEAK_INTENSITY < {max_intensity})\n')
    return cluster_centers
def sample_mono(mono_rays, sample_size):
    filtered = mono_rays[mono_rays[:,-1].argsort()][:sample_size]
    return filtered

def condensed_to_squareform_indices(condensed_index, n):
    # This function returns the pair of indices (i, j) in the square form matrix
    i = 0
    while condensed_index >= n - i - 1:
        condensed_index -= (n - i - 1)
        i += 1
    j = condensed_index + i + 1
    return i, j
def find_rot_mx(spot1_cart, spot2_cart, spot3_cart, spot4_cart):
    
    a = np.array([spot1_cart, spot2_cart])
    b = np.array([spot3_cart, spot4_cart])
    rot, rssd = R.align_vectors(a, b)
    return rot

if __name__ =='__main__':

    path_expt = r'testdata\dark_01__expt.h5'
    path_mono = r'testdata\Rh1_mono.h5'
    hkls_equiv = ((-1, -1, -1), (-1, 1, -1), (1, -1, 1))

    mono_sample_size = 300                      #number of highest intensity reflections taken from mono file for matching
    expt_min_intensity = 100                    #minimum intensity of expt reflections taken into clustering
    expt_max_intensity = 20000                  #maximum intensity of expt reflections taken into clustering
    cluster_complete_linkage_crit = 0.015       #angular distance parameter for complete hierarchical bottom-up clustering of expt peaks (radians)
    cluster_nb_of_peaks_crit = 5                #minimum number of reflections in a cluster, for an entire cluster to be considered for matching
    cluster_min_intensity = 1000                #minimum intensity of reflections in a cluster, for an entire cluster to be considered for matching
    cluster_max_intensity = 20000               #maximum intensity of reflections in a cluster, for an entire cluster to be considered for matching
    distance_match_threshold = 0.00001          #angular distance criterium for a successful match of mono and expt cluster pairs (radians)
    cluster_crit_rot = 0.02                     #angular distance parameter for complete hierarchical bottom-up clustering of orientation matrices (radians)
    cluster_nb_rot_crit = 10                     #minimum number of oreintation matrices in a cluster

    mono_cart, mono_sph, mono_int = import_mono(path_mono, hkls_equiv)
    mono_rays_cart = np.concatenate((mono_cart, mono_int[:, np.newaxis]), axis = 1)
    mono_rays_sph = np.concatenate((mono_sph, mono_int[:, np.newaxis]), axis = 1)

    print(f'\n{len(mono_cart)} monochromatic peaks in the mono file')

    mono_rays_sample_cart = sample_mono(mono_rays_cart, mono_sample_size)
    mono_sample_cart = mono_rays_sample_cart[:,:3]

    print(f'{mono_sample_size} monochromatic peaks sampled from the mono file (highest intensity criterium)\n')

    expt_cart, expt_sph, expt_int = import_expt(path_expt)
    expt_rays_cart = np.concatenate((expt_cart, expt_int[:, np.newaxis]), axis = 1)
    expt_rays_sph = np.concatenate((expt_sph, expt_int[:, np.newaxis]), axis = 1)

    print(f'{len(expt_cart)} expt peaks in the raw expt file')

    #initial intensity filter
    expt_rays_cart_filtered = np.array([ray for ray in expt_rays_cart if ray[3] > expt_min_intensity and ray[3] < expt_max_intensity])
    expt_rays_sph_filtered = np.array([ray for ray in expt_rays_sph if ray[2] > expt_min_intensity and ray[2] < expt_max_intensity])

    #print peaks left after initial filtering
    print(f'{len(expt_rays_cart_filtered)} expt peaks left after initial filtering: {expt_min_intensity} < PEAK_INTENSITY < {expt_max_intensity}\n')

    #cluster
    cluster_centers = cluster_hier(rays_cart=expt_rays_cart_filtered, rays_sph=expt_rays_sph_filtered, cluster_crit=cluster_complete_linkage_crit, cluster_nb_crit=cluster_nb_of_peaks_crit, min_intensity=cluster_min_intensity, max_intensity=cluster_max_intensity)

    draw_sphere(mono_cart, r'imgs/mono.png')
    draw_sphere(expt_cart, r'imgs/expt.png')
    draw_sphere(cluster_centers, r'imgs/expt_clustered.png')
    draw_sphere(mono_rays_sample_cart, r'imgs/mono_sampled.png')

    cluster_dist_mx_cart = scipy.spatial.distance.pdist(cluster_centers)
    cluster_dist_mx_sph = convert_distance_to_spherical(cluster_dist_mx_cart)
    del cluster_dist_mx_cart

    mono_dist_mx_cart = scipy.spatial.distance.pdist(mono_sample_cart)
    mono_dist_mx_sph = convert_distance_to_spherical(mono_dist_mx_cart)
    del mono_dist_mx_cart

    dif_dist = scipy.spatial.distance.cdist(cluster_dist_mx_sph[:, np.newaxis], mono_dist_mx_sph[:, np.newaxis])

    del cluster_dist_mx_sph
    del mono_dist_mx_sph

    hits_array = np.argwhere(dif_dist < distance_match_threshold)

    del dif_dist

    print (f'{len(hits_array)} similar distances found on both spheres ({distance_match_threshold} rad match threshold)')
    rot_mxs = []
    for x in hits_array:

        dist_matched_cluster_ind = x[0]
        dist_matched_mono_ind = x[1]

        cluster_total_obs_nb = len(cluster_centers)
        mono_total_obs_nb = len(mono_sample_cart)

        cluster_spot_sq_ind = condensed_to_squareform_indices(condensed_index=dist_matched_cluster_ind, n=cluster_total_obs_nb)
        mono_spot_sq_ind = condensed_to_squareform_indices(condensed_index=dist_matched_mono_ind, n=mono_total_obs_nb)

        spot_expt1, spot_expt2 = cluster_centers[cluster_spot_sq_ind[0]], cluster_centers[cluster_spot_sq_ind[1]]
        spot_mono1, spot_mono2 = mono_sample_cart[mono_spot_sq_ind[0]], mono_sample_cart[mono_spot_sq_ind[1]]

        rot = find_rot_mx(spot_expt1, spot_expt2, spot_mono1, spot_mono2)
        rot_mxs.append(rot)

    sample_vector = [0, 0, 1]
    final_vecs = []
    for rot in rot_mxs:
        vec = rot.apply(sample_vector)
        vec = vec / np.linalg.norm(vec)
        final_vecs.append(vec)

    final_vecs = np.array(final_vecs)
    draw_sphere(final_vecs, r'imgs/oms.png')

    rot_dist_mx_cart = scipy.spatial.distance.pdist(final_vecs)
    rot_dist_mx_sph = convert_distance_to_spherical(rot_dist_mx_cart)
    Z = scipy.cluster.hierarchy.complete(rot_dist_mx_sph)
    clustered = scipy.cluster.hierarchy.fcluster(Z, cluster_crit_rot, criterion='distance')
    cluster_nb = max(clustered)

    print (f'{cluster_nb} rot matrix clusters were found with clustering distance criterion of d < {cluster_crit_rot} rad (complete hierarchy linkage)')

    rot_cluster_centers = []
    for i in range(1, cluster_nb + 1):
        cluster_points = final_vecs[clustered == i]
        if len(cluster_points) < cluster_nb_rot_crit:
            continue
        theta_mean = np.mean(cluster_points[:,0])
        phi_mean = np.mean(cluster_points[:,1])
        cluster_carts = spherical_to_cartesian(theta_mean, phi_mean)
        rot_cluster_centers.append(cluster_carts)
    rot_cluster_centers = np.array(rot_cluster_centers)

    print (f'{len(rot_cluster_centers)} rot matrix clusters are left after filtering (at least {cluster_nb_rot_crit} rotM/cluster)\n')

    draw_sphere(rot_cluster_centers, r'imgs/rot_clustered.png')