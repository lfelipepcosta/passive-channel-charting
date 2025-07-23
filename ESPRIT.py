import numpy as np

def esprit_implementation(covariance_matrix, num_antennas, num_sources, normalized_spacing):
    """
    Direct implementation of the ESPRIT algorithm using only NumPy.
    This function estimates the Direction of Arrival (DoA) for signals
    arriving at a Uniform Linear Array (ULA).

    Args:
        covariance_matrix (np.ndarray): The covariance matrix of the received signal.
        num_antennas (int): The number of antennas in the ULA.
        num_sources (int): The number of signal sources to estimate.
        normalized_spacing (float): The spacing between antennas, normalized by the
                                     signal's wavelength (d/λ).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array containing the estimated angles of arrival, in radians.
            - np.ndarray: An array containing the eigenvalue magnitudes (confidence score).
    """
    # Step 1: Perform eigen-decomposition of the covariance matrix.
    # This breaks the matrix down into its fundamental components (eigenvalues and eigenvectors).
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Step 2: Sort eigenvectors to separate the signal and noise subspaces.
    # We sort the eigenvalues in descending order and get their original indices.
    sorted_indices = eigenvalues.argsort()[::-1]
    # The signal subspace is formed by the eigenvectors corresponding to the largest eigenvalues.
    signal_subspace = eigenvectors[:, sorted_indices[:num_sources]]

    # Step 3: Create two overlapping subarrays from the signal subspace.
    # This is the core principle of ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques).
    # Subarray 1 uses the first (n-1) antennas.
    subarray1 = signal_subspace[:-1, :]
    # Subarray 2 uses the last (n-1) antennas.
    subarray2 = signal_subspace[1:, :]

    # Step 4: Solve the matrix equation `subarray1 * Psi = subarray2` for the rotational matrix `Psi`.
    # The pseudo-inverse is used to find the least-squares solution, which is robust to noise.
    psi = np.linalg.pinv(subarray1) @ subarray2

    # Step 5: The eigenvalues of the rotational matrix `Psi` contain the phase information.
    psi_eigenvalues = np.linalg.eigvals(psi)

    # Step 6: Calculate the arrival angles from the phase of Psi's eigenvalues.
    # The phase of the eigenvalue is directly proportional to the sine of the arrival angle.
    # Formula: angle = arcsin(phase_of_eigenvalue / (2 * pi * normalized_spacing))
    angles_rad = np.arcsin(np.angle(psi_eigenvalues) / (2 * np.pi * normalized_spacing))
    
    # Step 7: Extract the magnitude of the eigenvalues as a confidence metric.
    # In a noiseless scenario, these should be 1. Deviation from 1 indicates noise.
    magnitudes = np.abs(psi_eigenvalues)

    return angles_rad, magnitudes