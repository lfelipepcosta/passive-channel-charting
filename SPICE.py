import numpy as np
from numpy.linalg import inv, norm

def steering_vector(theta_rad, M, d, c, f_c):
    """
    Generates the steering vector for a given angle.
    """
    k = 2 * np.pi * f_c / c
    m = np.arange(M)
    return np.exp(-1j * k * d * m * np.sin(theta_rad))

def estimate_spice_from_R(R_hat, M, d, c, f_c, num_peaks=1):
    """
    Estimates the Direction of Arrival (DOA) using a more robust implementation
    of the SPICE algorithm from a pre-computed spatial covariance matrix.

    Args:
        R_hat (np.ndarray): The measured (M, M) spatial covariance matrix.
        M (int): Number of antennas in the array.
        d (float): Spacing between sensors (in meters).
        c (float): Speed of light (in m/s).
        f_c (float): The center frequency of the signal (in Hz).
        num_peaks (int): The number of top angles (peaks) to return.

    Returns:
        tuple: A tuple containing:
            - estimated_angles_rad (np.ndarray): Array of estimated angles in radians.
            - peak_powers_db (np.ndarray): Array of the power of the found peaks in dB.
    """
    # 1. Define the angular grid and construct the array manifold matrix 'A'
    angles_deg = np.linspace(-90, 90, 361)
    theta_grid_rad = np.deg2rad(angles_deg)
    A = np.stack([steering_vector(theta, M, d, c, f_c) for theta in theta_grid_rad], axis=1)
    
    # Add diagonal loading to the measured covariance matrix for stability
    R_hat_reg = R_hat + 1e-5 * np.eye(M)

    # 2. SPICE iterative algorithm implementation
    n_angles = A.shape[1]
    p = np.ones(n_angles) * (norm(R_hat_reg, 'fro') / n_angles)
    max_iter = 100
    tol = 1e-6

    for i in range(max_iter):
        p_old = p.copy()

        # This improved update rule is more faithful to the original SPICE algorithm's
        # goal of minimizing the covariance fitting criterion.
        
        # a. Build the theoretical covariance matrix from current power estimates
        R_current_model = A @ np.diag(p) @ A.conj().T
        
        # b. Invert the current model's covariance matrix
        try:
            R_inv = inv(R_current_model + 1e-5 * np.eye(M))
        except np.linalg.LinAlgError:
            return np.array([]), np.array([]) # Return empty if inversion fails

        # c. Calculate weights for the update
        # P_k measures how much power the measured data projects onto each steering vector direction
        # w_k is a normalization factor
        P_k = np.real(np.einsum('ij,ji->i', A.conj().T, R_inv @ R_hat_reg @ R_inv @ A))
        w_k = np.real(np.einsum('ij,ji->i', A.conj().T, R_inv @ A))
        
        # d. Update the power estimates
        p = p_old * np.sqrt(P_k / (w_k + 1e-12))

        # e. Check for convergence
        if norm(p - p_old) < tol * norm(p_old):
            # print(f"Converged after {i+1} iterations.")
            break

    P_spice = p
    
    # 3. Find peaks in the final power spectrum
    if np.all(P_spice == 0) or np.all(np.isnan(P_spice)) or np.max(P_spice) == 0:
        return np.array([]), np.array([])
        
    P_final_db = 10 * np.log10(P_spice / np.max(P_spice))
    
    from scipy.signal import find_peaks
    peak_indices, properties = find_peaks(P_final_db, height=-30, distance=10)
    
    if len(peak_indices) == 0:
        return np.array([]), np.array([])
    
    peak_heights = properties['peak_heights']
    top_indices_sorted = peak_indices[np.argsort(peak_heights)[::-1][:num_peaks]]
    
    estimated_angles_deg = angles_deg[top_indices_sorted]
    peak_powers_db = P_final_db[top_indices_sorted]
    
    sort_order = np.argsort(estimated_angles_deg)
    estimated_angles_deg_sorted = estimated_angles_deg[sort_order]
    peak_powers_db_sorted = peak_powers_db[sort_order]

    estimated_angles_rad = np.deg2rad(estimated_angles_deg_sorted)

    return estimated_angles_rad, peak_powers_db_sorted