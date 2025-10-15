# DELAY_AND_SUM.py
import numpy as np
from scipy.signal import find_peaks

def estimate_das_from_R(R, M, d, c, f_c, num_peaks=1):
    """
    Estimates the Direction of Arrival (DOA) using the Delay-and-Sum (DAS)
    beamformer from a pre-computed spatial covariance matrix R.

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
    angles_deg = np.linspace(-90, 90, 361)
    # Initialize an array to store the DAS power spectrum.
    P_das = np.zeros(len(angles_deg))

    # --- Angle Scanning Loop: Iterate over each possible angle ---
    for i_theta, theta in enumerate(angles_deg):
        # Convert the current test angle to radians.
        theta_rad = np.deg2rad(theta)
        # Construct the steering vector a(theta) for this angle.
        steering_vector = np.exp(-1j * 2 * np.pi * f_c * d * np.arange(M) * np.sin(theta_rad) / c)
        steering_vector = steering_vector[:, np.newaxis] # Reshape to a column vector

        # Calculate the DAS power: P(theta) = a(theta)^H * R * a(theta).
        power = np.dot(steering_vector.conj().T, np.dot(R, steering_vector))
        
        # Store the calculated power for this angle.
        P_das[i_theta] = np.abs(power[0, 0])

    # --- Final Spectrum Peak Finding ---
    # Convert the final spectrum to dB.
    P_final_db = 10 * np.log10(P_das / np.max(P_das))
    
    # Find peaks (angles) in the final power spectrum.
    peak_indices, properties = find_peaks(P_final_db, height=-40, distance=10)
    
    if len(peak_indices) == 0:
        return np.array([]), np.array([])
    
    # Sort the found peaks by their power and select the top ones.
    peak_heights = properties['peak_heights']
    top_indices_sorted_by_height = peak_indices[np.argsort(peak_heights)[::-1][:num_peaks]]
    
    estimated_angles_deg = angles_deg[top_indices_sorted_by_height]
    peak_powers_db = P_final_db[top_indices_sorted_by_height]
    
    # Sort final results by angle for consistency.
    sort_order = np.argsort(estimated_angles_deg)
    estimated_angles_deg_sorted = estimated_angles_deg[sort_order]
    peak_powers_db_sorted = peak_powers_db[sort_order]

    # Convert final angles to radians.
    estimated_angles_rad = np.deg2rad(estimated_angles_deg_sorted)

    return estimated_angles_rad, peak_powers_db_sorted