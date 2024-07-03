import numpy as np
from scipy.fftpack import ifft2
import matplotlib.pyplot as plt

def spherical_to_cartesian(theta, phi):
    """Convert spherical coordinates to Cartesian coordinates."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def cartesian_to_spherical(points):
    """Convert Cartesian coordinates to spherical coordinates."""
    r = np.linalg.norm(points, axis=1)
    theta = np.arccos(points[:, 2] / r)
    phi = np.arctan2(points[:, 1], points[:, 0])
    return theta, phi

def create_histogram(theta, phi, bins=64):
    """Create a 2D histogram of spherical coordinates."""
    hist, xedges, yedges = np.histogram2d(theta, phi, bins=bins, range=[[0, np.pi], [0, 2 * np.pi]])
    return hist

def fft_correlation_2d(pattern1, pattern2, bins=128):
    # Convert Cartesian coordinates to spherical coordinates
    theta1, phi1 = cartesian_to_spherical(pattern1)
    theta2, phi2 = cartesian_to_spherical(pattern2)
    
    # Create 2D histograms
    hist1 = create_histogram(theta1, phi1, bins)
    hist2 = create_histogram(theta2, phi2, bins)
    
    # Compute 2D FFT
    fft1 = np.fft.fft2(hist1)
    fft2 = np.fft.fft2(hist2)
    
    # Compute cross-correlation using FFT
    fft_product = fft1 * np.conj(fft2)
    correlation = ifft2(fft_product)
    
    # Find the peak in the correlation map
    max_corr = np.abs(correlation).max()
    peak_coords = np.unravel_index(np.argmax(np.abs(correlation)), correlation.shape)
    
    return max_corr, peak_coords, hist1, hist2, correlation

# Example usage with random points
np.random.seed(0)
pattern1 = np.random.randn(20000, 3)
pattern2 = np.random.randn(13000, 3)

# Normalize to unit vectors (unit sphere)
pattern1 /= np.linalg.norm(pattern1, axis=1, keepdims=True)
pattern2 /= np.linalg.norm(pattern2, axis=1, keepdims=True)

correlation, peak_coords, hist1, hist2, correlation_map = fft_correlation_2d(pattern1, pattern2, bins=64)
print(f"FFT Correlation: {correlation}")
print(f"Peak Coordinates: {peak_coords}")

# Visualization of the histograms and correlation map
plt.figure(figsize=(18, 6))

plt.subplot(131)
plt.title('Histogram of Pattern 1')
plt.imshow(hist1.T, origin='lower', extent=[0, np.pi, 0, np.pi], aspect='auto')
plt.xlabel('Theta')
plt.ylabel('Phi')

plt.subplot(132)
plt.title('Histogram of Pattern 2')
plt.imshow(hist2.T, origin='lower', extent=[0, np.pi, 0, np.pi], aspect='auto')
plt.xlabel('Theta')
plt.ylabel('Phi')

plt.subplot(133)
plt.title('Correlation Map')
plt.imshow(np.abs(correlation_map).T, origin='lower', extent=[0, np.pi, 0, 2 * np.pi], aspect='auto')
plt.xlabel('Theta')
plt.ylabel('Phi')

plt.tight_layout()
plt.show()
