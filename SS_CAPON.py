# SS_CAPON.py
import numpy as np
# We import our existing CAPON module to reuse its code.
import CAPON

def estimate_ss_capon(R, M, m, d, c, f_c, num_peaks=1):
    """
    Performs Spatial Smoothing on the covariance matrix R and then applies
    the Capon algorithm.

    Args:
        R (np.ndarray): The original, full-size (M, M) covariance matrix.
        M (int): The total number of antennas in the full array.
        m (int): The size of the subarrays to be used for smoothing.
        d (float): Spacing between sensors (in meters).
        c (float): Speed of light (in m/s).
        f_c (float): The center frequency of the signal (in Hz).
        num_peaks (int): The number of top angles (peaks) to return.

    Returns:
        tuple: The output from the Capon estimator (angles_rad, powers_db).
    """
    # 1. Determine the number of overlapping subarrays (L).
    L = M - m + 1
    
    # 2. Initialize the smoothed covariance matrix with the correct subarray size.
    R_ss = np.zeros((m, m), dtype=np.complex64)
    
    # 3. Sum the covariance matrices of all subarrays.
    for i in range(L):
        # Slice the original R to get the covariance matrix of the i-th subarray.
        subarray_R = R[i:i+m, i:i+m]
        R_ss += subarray_R
        
    # 4. Average the result. This is the final smoothed covariance matrix.
    R_ss /= L
    
    # 5. Feed the smoothed matrix (R_ss) into our existing Capon estimator.
    #    Note that the number of antennas passed to the Capon function is now 'm' (subarray size).
    estimated_angles_rad, peak_powers_db = CAPON.estimate_capon_from_R(
        R_ss, m, d, c, f_c, num_peaks
    )
    
    return estimated_angles_rad, peak_powers_db