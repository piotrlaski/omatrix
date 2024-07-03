import h5py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy

from functools import wraps
from time import time

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

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return [r, theta, phi]

def draw_mono():

    with h5py.File(path_mono, 'r') as mono:
        h_mono = np.array(mono['mono']['h_unit'])

    vecs = h_mono
    for equiv in hkls_equiv:
        new_hkls = h_mono * equiv
        vecs = np.concatenate((vecs, new_hkls))

    U, V, W = zip(*vecs)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(U, V, W, s=1, marker = '.', lw = 0)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plt.savefig('mono.png', dpi = 1200)

def draw_expt():

    with h5py.File(path_expt, 'r') as expt:
        h_expt = np.array(expt['expt']['h_unit_g'])

    vecs = h_expt

    U, V, W = zip(*vecs)
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

    plt.savefig('expt.png', dpi = 1200)

def draw_expt_and_fill():

    with h5py.File(path_expt, 'r') as expt:
        h_expt = np.array(expt['expt']['h_unit_g'])

    vecs = h_expt
    for equiv in hkls_equiv:
        new_hkls = h_expt * equiv
        vecs = np.concatenate((vecs, new_hkls))

    U, V, W = zip(*vecs)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(U, V, W, s=1, marker = '.', lw = 0)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plt.savefig('expt_filled.png', dpi = 1200)
    plt.show()

def npfft(vecs_sph):
    return np.fft.rfft2(vecs_sph)


# draw_mono()
# draw_expt()
#draw_expt_and_fill()

with h5py.File(path_mono, 'r') as mono:
    h_mono = np.array(mono['mono']['h_unit'])

vecs = h_mono
for equiv in hkls_equiv:
    new_hkls = h_mono * equiv
    vecs = np.concatenate((vecs, new_hkls))

vecs_sph = []
for vec in vecs:
    vec_sph = cartesian_to_spherical(*vec)
    vecs_sph.append(vec_sph[1:])

vecs_sph = np.array(vecs_sph)
# t_vecs = torch.from_numpy(vecs_sph)

numpyft = npfft(vecs_sph)
# t_vecs.to(device)
# torchft = ptfft(t_vecs)

v_mono = vecs_sph

with h5py.File(path_mono, 'r') as expt:
    h_expt = np.array(expt['mono']['h_unit'])
# with h5py.File(path_expt, 'r') as expt:
#     h_expt = np.array(expt['expt']['h_unit_g'])

vecs = h_expt
vecs_sph = []
for vec in vecs:
    vec_sph = cartesian_to_spherical(*vec)
    vecs_sph.append(vec_sph[1:])

vecs_sph = np.array(vecs_sph)
numpyft2 = npfft(vecs_sph)

v_expt = vecs_sph

corr = scipy.signal.correlate(v_mono, v_expt, method = 'fft', mode = 'valid')

plt.plot(corr)
plt.savefig('test.png')
pass