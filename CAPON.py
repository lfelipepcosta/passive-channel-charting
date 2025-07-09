import numpy as np
from scipy.signal import find_peaks

def estimate_capon_from_R(R, M, d, c, f_c, num_peaks=1):
    """
    Estimates the Direction of Arrival (DOA) using the Capon algorithm
    from a pre-computed spatial covariance matrix R.

    Args:
        R (np.ndarray): The (M, M) spatial covariance matrix.
        M (int): Number of sensors (antennas) in the array.
        d (float): Spacing between sensors (in meters).
        c (float): Speed of light (in m/s).
        f_c (float): The center frequency of the signal (in Hz).
        num_peaks (int): The number of top angles (peaks) to return.

    Returns:
        tuple: A tuple containing:
            - estimated_angles_rad (np.ndarray): Array of estimated angles in radians.
            - peak_powers_db (np.ndarray): Array of the power of the found peaks in dB.
    """
    # Define the grid of angles to scan, from -90 to +90 degrees.
    angles_deg = np.linspace(-90, 90, 361) # Increased resolution for better peaks
    # Initialize an array to store the Capon spectrum.
    P_capon = np.zeros(len(angles_deg))

    # Apply diagonal loading for regularization, ensuring R is invertible and stable.
    R_reg = R + 1e-5 * np.eye(M)

    # --- Angle Scanning Loop: Iterate over each possible angle ---
    for i_theta, theta in enumerate(angles_deg):
        # Convert the current test angle to radians.
        theta_rad = np.deg2rad(theta)
        # Construct the steering vector a(theta) for this angle and the center frequency.
        steering_vector = np.exp(-1j * 2 * np.pi * f_c * d * np.arange(M) * np.sin(theta_rad) / c)
        steering_vector = steering_vector[:, np.newaxis] # Reshape to a column vector

        # Calculate the Capon spectrum denominator: a(theta)^H * R_inv * a(theta).
        # np.linalg.solve(R_reg, steering_vector) is an efficient way to compute R_inv * a(theta).
        denom = np.dot(steering_vector.conj().T, np.linalg.solve(R_reg, steering_vector))
        
        # The Capon power is the reciprocal of the denominator.
        P_capon[i_theta] = 1 / np.abs(denom[0, 0])

    # --- Final Spectrum Peak Finding ---
    # Convert the final spectrum to dB for easier visualization and peak finding.
    P_final_db = 10 * np.log10(P_capon / np.max(P_capon))

    # Find peaks (angles) in the final power spectrum.
    peak_indices, properties = find_peaks(P_final_db, height=-40, distance=10)
    
    # If no peaks are found, return empty arrays.
    if len(peak_indices) == 0:
        return np.array([]), np.array([])
    
    # Sort the found peaks by their power (height) and select the top 'num_peaks'.
    peak_heights = properties['peak_heights']
    top_indices_sorted_by_height = peak_indices[np.argsort(peak_heights)[::-1][:num_peaks]]
    
    # Get the angles in degrees corresponding to the top peaks.
    estimated_angles_deg = angles_deg[top_indices_sorted_by_height]
    # Get the power of the found peaks.
    peak_powers_db = P_final_db[top_indices_sorted_by_height]
    
    # Sort the final results by angle for consistency.
    sort_order = np.argsort(estimated_angles_deg)
    estimated_angles_deg_sorted = estimated_angles_deg[sort_order]
    peak_powers_db_sorted = peak_powers_db[sort_order]

    # Convert final angles to radians for compatibility with the rest of the pipeline.
    estimated_angles_rad = np.deg2rad(estimated_angles_deg_sorted)

    return estimated_angles_rad, peak_powers_db_sorted