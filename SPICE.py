import numpy as np
from numpy.linalg import inv, norm
from scipy.signal import find_peaks
from scipy.linalg import sqrtm

def steering_vector(theta_rad, M, d, c, f_c):
    """
    Generates the steering vector for a given angle.
    """
    k = 2 * np.pi * f_c / c
    m = np.arange(M)
    return np.exp(-1j * k * d * m * np.sin(theta_rad))

def estimate_spice_from_R(R_hat, M, d, c, f_c, num_peaks=1):
    """
    Estimates the Direction of Arrival (DOA) using the standard iterative
    SPICE algorithm from a pre-computed spatial covariance matrix.

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
    # 1. Setup: Define angular grid and build the array manifold matrix 'A'
    angles_deg = np.linspace(-90, 90, 361) # Grade de busca
    theta_grid_rad = np.deg2rad(angles_deg)
    A = np.stack([steering_vector(theta, M, d, c, f_c) for theta in theta_grid_rad], axis=1) # Matriz com os steering vectors para todas as direções de busca
    
    # Add diagonal loading for numerical stability
    R_hat_reg = R_hat + 1e-6 * np.eye(M)

    # Calculate the matrix square root of R_hat once, outside the loop.
    # We use sqrtm for the matrix square root.
    R_hat_sqrt = sqrtm(R_hat_reg) # Cálculo da raiz quadrada matricial (pré processamento para SPICE)

    # 2. Initialization
    # Initialize powers using a simple beamformer output for faster convergence
    initial_beamformer_spectrum = np.einsum('ij,ji->i', A.conj().T, R_hat_reg @ A) # Cálculo da potência inicial com um DAS (calculando a potência para todas as direções)
    # Fórmula acima: p(theta) = a(theta)ˆH R_chapeu a(theta)
    p = np.abs(initial_beamformer_spectrum) # Converte o valor real para potência (magnitude), garantindo um valor real e positivo
    
    max_iter = 100
    tol = 1e-6

    # 3. Main SPICE Iteration Loop
    for i in range(max_iter):
        p_old = p.copy() # Guarda a estimativa de potência atual para comparar no final da iteração

        # Adding a small noise term to the model is good practice for stability
        # and aligns better with the theoretical model R = APA^H + sigma*I
        noise_power = np.mean(p) * 1e-4 # A small noise estimate
        R_model = A @ np.diag(p) @ A.conj().T + noise_power * np.eye(M) # Matriz MxM que tem como seria a matriz de covariância se as potências calculadas estivessem certas
        R_model_inv = inv(R_model) # Cálculo da inversa da matriz modelo para poder usar posteriormente

        # The formula is based on Equation (14) from the user-provided paper.
        # We calculate the term: A^H * R_inv * R_hat_sqrt
        update_matrix = A.conj().T @ R_model_inv @ R_hat_sqrt # # Matriz 361xM onde cada linha é a "evidência" de sinal inexplicado para a direção correspondente
        
        # The update factor is the Frobenius norm of each row of the resulting matrix.
        # np.linalg.norm(..., axis=1) computes the norm for each row individually.
        update_factor = norm(update_matrix, axis=1) # Cálculo da norma de cada direção em um único número de "força" (quanto maior, maior a evidência de um sinal inexplicado vindo daquela direção)
        
        # The update is multiplicative
        p = p_old * update_factor # Multiplicação elemento a elemento para evidenciar os ângulos com mais "força" (passo de aprendizado)
        
        # Normalization (optional, but helps with stability)
        if np.sum(p) > 0:
            p = p / np.sum(p) # Normaliza o vetor para focar nas proporções relativas e garantir estabilidade numérica

        # Check for convergence
        if norm(p - p_old) < tol * norm(p_old): # Se a mudança nas potências for menor que a tolerância, ou seja, praticamente não mudar, o algoritmo encerra
            # print(f"Converged after {i+1} iterations.")
            break

    # 4. Find peaks in the final power spectrum
    P_spice = p
    
    if np.all(P_spice == 0) or np.all(np.isnan(P_spice)) or np.max(P_spice) == 0:
        return np.array([]), np.array([])
        
    # Use the actual power for the spectrum, not a normalized version
    P_final_db = 10 * np.log10(P_spice)
    
    # Find peaks. A height relative to the max peak is often more robust.
    peak_indices, properties = find_peaks(P_final_db, height=np.max(P_final_db)-20, distance=10)
    
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