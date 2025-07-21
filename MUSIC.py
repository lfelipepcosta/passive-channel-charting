import numpy as np
from scipy.signal import find_peaks

def estimate_music_from_R(R, M, d, c, f_c, num_sources=1, num_peaks=1):
    """
    Estimates the Direction of Arrival (DOA) using the standard MUSIC algorithm
    from a pre-computed spatial covariance matrix R.

    Args:
        R (np.ndarray): The (M, M) spatial covariance matrix.
        M (int): Number of sensors (antennas) in the array.
        d (float): Spacing between sensors (in meters).
        c (float): Speed of light (in m/s).
        f_c (float): The center frequency of the signal (in Hz).
        num_sources (int): The number of signal sources to estimate.
        num_peaks (int): The number of top angles (peaks) to return.

    Returns:
        tuple: A tuple containing:
            - estimated_angles_rad (np.ndarray): Array of estimated angles in radians.
            - peak_powers_db (np.ndarray): Array of the power of the found peaks in dB.
    """
    # 1. Perform eigen-decomposition of the covariance matrix.
    eigenvalues, eigenvectors = np.linalg.eig(R)
    
    # 2. Sort eigenvalues and eigenvectors in descending order.
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # 3. Isolate the noise subspace.
    # The noise subspace is formed by the eigenvectors corresponding to the
    # (M - num_sources) smallest eigenvalues.
    E_n = eigenvectors[:, num_sources:]
    
    # Define the grid of angles to scan.
    angles_deg = np.linspace(-90, 90, 361)
    P_music = np.zeros(len(angles_deg))

    # --- 4. Angle Scanning Loop ---
    for i_theta, theta in enumerate(angles_deg):
        theta_rad = np.deg2rad(theta)
        # Construct the steering vector a(theta) for this angle.
        steering_vector = np.exp(-1j * 2 * np.pi * f_c * d * np.arange(M) * np.sin(theta_rad) / c)
        steering_vector = steering_vector[:, np.newaxis]

        # Calculate the MUSIC spectrum denominator: a(theta)^H * E_n * E_n^H * a(theta).
        # This measures how "close" the steering vector is to the noise subspace.
        # When a(theta) is orthogonal to E_n, this value is close to zero.
        projection = np.dot(E_n.conj().T, steering_vector)
        denom = np.dot(projection.conj().T, projection)
        
        # The MUSIC power is the reciprocal of this value.
        P_music[i_theta] = 1 / np.abs(denom[0, 0])

    # --- 5. Final Spectrum Peak Finding ---
    P_final_db = 10 * np.log10(P_music / np.max(P_music))
    peak_indices, properties = find_peaks(P_final_db, height=-40, distance=10)
    
    if len(peak_indices) == 0:
        return np.array([]), np.array([])
    
    peak_heights = properties['peak_heights']
    top_indices_sorted_by_height = peak_indices[np.argsort(peak_heights)[::-1][:num_peaks]]
    
    estimated_angles_deg = angles_deg[top_indices_sorted_by_height]
    peak_powers_db = P_final_db[top_indices_sorted_by_height]
    
    sort_order = np.argsort(estimated_angles_deg)
    estimated_angles_deg_sorted = estimated_angles_deg[sort_order]
    peak_powers_db_sorted = peak_powers_db[sort_order]

    estimated_angles_rad = np.deg2rad(estimated_angles_deg_sorted)

    return estimated_angles_rad, peak_powers_db_sorted