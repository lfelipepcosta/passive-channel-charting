from tqdm.auto import tqdm
import espargos_0007
import cluster_utils
import numpy as np
import CRAP
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import time
import sys

from sklearn.decomposition import PCA

# Get the round number from the command-line arguments
round_num = sys.argv[1] if len(sys.argv) > 1 else '1'

DEBUG = False

# Data Loading and Preprocessing

# Loading all the datasets can take some time...
training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)

all_datasets = training_set_robot + test_set_robot + test_set_human

# Load the pre-computed clutter signatures for each dataset
for dataset in all_datasets:
    dataset['clutter_acquisitions'] = np.load(os.path.join("clutter_channel_estimates", os.path.basename(dataset['filename']) + ".npy"))

# Create a directory to store the output AoA estimates
os.makedirs("aoa_estimates", exist_ok=True)
os.makedirs("aoa_estimates_PCA", exist_ok=True)

# Group the data into temporal clusters
for dataset in all_datasets:
    cluster_utils.cluster_dataset(dataset)

# Unitary Root-MUSIC Algorithm Definition
def get_unitary_rootmusic_estimator(chunksize = 32, shed_coeff_ratio = 0):  # `chunksize` is the number of antennas in the array
    
    # The Q matrix is a special Unitary Transformation Matrix. Its purpose is to convert the complex-valued covariance matrix R into a new, real-valued
    # matrix C via the operation: C = real(Q.H @ R @ Q).
    #
    # This transformation has two main benefits for Angle of Arrival (AoA) estimation:
    # 1. It improves performance when dealing with correlated signals, which are common in environments with multipath reflections.
    # 2. It effectively doubles the amount of data ("forward-backward averaging"), leading to a more robust estimate with fewer samples.

    I = np.eye(chunksize // 2)
    J = np.flip(np.eye(chunksize // 2), axis = -1)
    Q = np.asmatrix(np.block([[I, 1.0j * I], [J, -1.0j * J]]) / np.sqrt(2))
    
    def unitary_rootmusic(R):
        """
        The actual MUSIC estimator function.
        """
        assert(len(R) == chunksize)
        # Apply the unitary transformation to the covariance matrix R
        C = np.real(Q.H @ R @ Q)

        # Perform eigen-decomposition on the transformed matrix
        eig_val, eig_vec = np.linalg.eigh(C)
        eig_val = eig_val[::-1]
        eig_vec = eig_vec[:,::-1]

        # Separate the noise subspace (En), which is formed by the eigenvectors corresponding to the smallest eigenvalues. We assume 1 signal source
        source_count = 1
        En = eig_vec[:,source_count:]
        # Construct the polynomial matrix from the noise subspace and unitary matrix
        ENSQ = Q @ En @ En.T @ Q.H

        # Calculate the polynomial coefficients by summing the diagonals of the ENSQ matrix.
        # np.trace with an offset sums a specific diagonal
        coeffs = np.asarray([np.trace(ENSQ, offset = diag) for diag in range(1, len(R))])
        coeffs = coeffs[:int(len(coeffs) * (1 - shed_coeff_ratio))]
        
        # Assemble the full, conjugate-symmetric polynomial from the coefficients
        coeffs = np.hstack((coeffs[::-1], np.trace(ENSQ), coeffs.conj()))
        # Find the roots of the polynomial
        roots = np.roots(coeffs)
        # Keep only the roots inside the unit circle
        roots = roots[abs(roots) < 1.0]
        # Find the root closest to the unit circle, which corresponds to the AoA
        largest_root = np.argmax(1 / (1.0 - np.abs(roots)))
        
        # Return the angle of the signal root (the AoA) and its magnitude (confidence)
        return np.angle(roots[largest_root]), np.abs(roots[largest_root])

    return unitary_rootmusic

# Create a MUSIC estimator for a 4-antenna array
umusic = get_unitary_rootmusic_estimator(4)

os.makedirs("subcarrieres_power_plots", exist_ok=True)

n_components_real_list = []
n_components_imag_list = []
energy_ratio_real_list = []
energy_ratio_imag_list = []
singular_values_real_list = []
singular_values_imag_list = []

start_time = time.perf_counter()
for dataset in tqdm(all_datasets):
    print(f"AoA estimation for dataset: {dataset['filename']}")

    # Initialize lists to store results for each cluster
    dataset['cluster_aoa_angles'] = []
    dataset['cluster_aoa_powers'] = []

    # Iterate through each cluster
    for cluster in tqdm(dataset['clusters']):
        # First, remove clutter from the CSI data for this cluster
        csi_by_transmitter_noclutter = []
        for tx_idx, csi in enumerate(cluster['csi_freq_domain']):
            csi_by_transmitter_noclutter.append(CRAP.remove_clutter(csi, dataset['clutter_acquisitions'][tx_idx]))

        if DEBUG:
            # Print how many CSI tensors (one per transmitter) we have in this cluster.
            print(f"\n[DEBUG] Cluster contains {len(csi_by_transmitter_noclutter)} CSI tensors (one per transmitter).")


        # Calculate the spatial covariance matrix R for each of the 4 receiver arrays
        R_old = np.zeros((espargos_0007.ARRAY_COUNT, espargos_0007.COL_COUNT, espargos_0007.COL_COUNT), dtype = np.complex64)
        R_array = np.zeros((espargos_0007.ARRAY_COUNT, espargos_0007.COL_COUNT, espargos_0007.COL_COUNT), dtype = np.complex64)

        for tx_csi in csi_by_transmitter_noclutter:

            # tx_csi é um tensor do NumPy com 5 dimensões (snapshot, array, linha, coluna, subportadora) 

            if DEBUG:
                print(f"[DEBUG] Shape of 'tx_csi' tensor: {tx_csi.shape}")
                print(f"[DEBUG] Value of one signal element: {tx_csi[0, 0, 0, 0, 0]}")

            # Average the covariance matrices from all transmitters
            R_old = R_old + (np.einsum("dbrms,dbrns->bmn", tx_csi, np.conj(tx_csi)) / tx_csi.shape[0])

            (num_snapshots, num_arrays, num_rows, num_antennas_per_array, num_subcarriers) = tx_csi.shape

            for snapshot_i in range(num_snapshots):

                for array_i in range(num_arrays):

                    # Fazer o reshape a partir daqui para esses 4 vetores (tem que considerar que tem que fazer a matriz de correlação)
                    # Com o vetor grande aplicar a PCA.transform, selecionar os componentes de maior energia, aplicar PCA.transform_inverse

                    # Slice the tensor to get data for the current snapshot and array
                    csi_for_single_array_snapshot = tx_csi[snapshot_i, array_i, :, :, :] # .shape = (2, 4, 53)

                    transposed_csi_for_single_array_snapshot = np.transpose(csi_for_single_array_snapshot, (2, 0, 1)) # .shape = (53, 2, 4)
                    
                    # Reshape the 3D tensor into a 2D matrix for preprocessing
                    reshaped_vector_for_single_array = transposed_csi_for_single_array_snapshot.reshape(num_subcarriers, num_rows * num_antennas_per_array) # .shape = (53, 8)

                    PLOT_SUBCARRIER_POWER = False
                    if PLOT_SUBCARRIER_POWER and snapshot_i < 5:
                        power_snapshot = np.abs(csi_for_single_array_snapshot)**2
                        mean_power_per_subcarrier = np.mean(power_snapshot, axis=(0, 1))
                        
                        plt.figure(figsize=(12, 6))
                        plt.plot(np.arange(num_subcarriers), 10 * np.log10(mean_power_per_subcarrier))
                        plt.title(f'Mean Power per Subcarrier (Array {array_i}, Snapshot {snapshot_i})')
                        plt.xlabel('Subcarrier Index')
                        plt.ylabel('Mean Power (dB)')
                        plt.grid(True)
                        plt.xticks(np.arange(0, num_subcarriers, 2))
                        
                        plots_output_dir = "subcarrieres_power_plots"
                        plot_filename = f"subcarrier_power_plot_array_{array_i}_snapshot_{snapshot_i}.png"

                        full_plot_path = os.path.join(plots_output_dir, plot_filename)
                        
                        plt.savefig(full_plot_path)
                        plt.close()

                    # Aplicar pré processamento em cima do vetor por array após o reshape
                    DROP_SUBCARRIERS = False
                    SUBCARRIERS_TO_DROP = [26] 
                    num_subcarriers_after_drop = num_subcarriers 

                    if DROP_SUBCARRIERS and len(SUBCARRIERS_TO_DROP) > 0:
                        # np.delete deletes all rows specified in the list
                        vector_after_drop = np.delete(reshaped_vector_for_single_array, SUBCARRIERS_TO_DROP, axis=0)
                        reshaped_vector_for_single_array = vector_after_drop

                        num_subcarriers_after_drop = num_subcarriers - len(SUBCARRIERS_TO_DROP)

                    # Usar um modelo pra parte real e outro pra parte imaginária
                    # Create the PCA object.
                    # pca_model = PCA(n_components=0.90) # Passing a float (0.0 to 1.0) tells PCA to select the number of components needed to explain that percentage of the variance (0.90 means "keep 90% of the energy")

                    # scikit-learn's PCA does not natively support complex numbers.
                    # The standard approach is to process the real and imaginary parts separately.
                    
                    # Separate the real and imaginary parts of the (53, 8) matrix
                    real_part = np.real(reshaped_vector_for_single_array)
                    imag_part = np.imag(reshaped_vector_for_single_array)

                    # Apply PCA to the real part
                    pca_model_real = PCA(n_components=0.90)
                    real_part_transformed = pca_model_real.fit_transform(real_part)
                    real_part_reconstructed = pca_model_real.inverse_transform(real_part_transformed)

                    # Apply PCA to the imaginary part
                    pca_model_imag = PCA(n_components=0.90)
                    imag_part_transformed = pca_model_imag.fit_transform(imag_part)
                    imag_part_reconstructed = pca_model_imag.inverse_transform(imag_part_transformed)

                    # Captura as estatísticas após o 'fit' de cada modelo
                    n_components_real_list.append(pca_model_real.n_components_)
                    n_components_imag_list.append(pca_model_imag.n_components_)
                    energy_ratio_real_list.append(np.sum(pca_model_real.explained_variance_ratio_))
                    energy_ratio_imag_list.append(np.sum(pca_model_imag.explained_variance_ratio_))
                    
                    # Captura a média dos valores singulares para esta iteração
                    if pca_model_real.singular_values_.size > 0:
                        singular_values_real_list.append(np.mean(pca_model_real.singular_values_))
                    if pca_model_imag.singular_values_.size > 0:
                        singular_values_imag_list.append(np.mean(pca_model_imag.singular_values_))
                    
                    # Recombine the real and imaginary parts to form the processed vector
                    processed_vector_for_single_array = real_part_reconstructed + 1j * imag_part_reconstructed

                    temp_reshaped = processed_vector_for_single_array.reshape(num_subcarriers_after_drop, num_rows, num_antennas_per_array)  # .shape = (53, 2, 4)

                    processed_csi_for_single_array = np.transpose(temp_reshaped, (1, 2, 0)) # .shape = (2, 4, 53)
                    
                    if DEBUG:
                        print(f"[DEBUG] Snapshot {snapshot_i}, Array {array_i}:")
                        print(f"[DEBUG]   - Shape before reshaping: {csi_for_single_array_snapshot.shape}")
                        print(f"[DEBUG]   - Shape after reshaping:  {reshaped_vector_for_single_array.shape}")
                        print(f"[DEBUG]   - Shape after reshaping back:   {processed_csi_for_single_array.shape}\n")

                    for row_i in range(num_rows):
                        for antenna_i in range(num_antennas_per_array):
                            for antenna_j in range(num_antennas_per_array):
                                for subcarrier_i in range(num_subcarriers_after_drop):

                                    single_array_raw_signal_i = processed_csi_for_single_array[row_i, antenna_i, subcarrier_i] # Criar um vetor por array (4 vetores) a partir daqui para aplicação da PCA/KPCA a nível de antenas
                                    single_array_raw_signal_j = processed_csi_for_single_array[row_i, antenna_j, subcarrier_i]
                                    R_array[array_i, antenna_i, antenna_j] = R_array[array_i, antenna_i, antenna_j] + ((single_array_raw_signal_i * np.conj(single_array_raw_signal_j))/num_snapshots)
  
            if DEBUG:
                difference_array = np.sum(R_old - R_array)
                print(f"[DEBUG] Sum of the difference between R_old and R_array: {difference_array}\n")

        # Apply the MUSIC algorithm to each of the 4 receiver arrays' covariance matrices
        music_results = [umusic(R_array[array]) for array in range(R_array.shape[0])]

        # Convert the electrical angle from MUSIC to a physical AoA in radians and store it
        dataset['cluster_aoa_angles'].append(np.asarray([np.arcsin(angle_power[0] / np.pi) for angle_power in music_results]))
        dataset['cluster_aoa_powers'].append(np.asarray([angle_power[1] for angle_power in music_results]))

    # Convert result lists to NumPy arrays
    dataset['cluster_aoa_angles'] = np.asarray(dataset['cluster_aoa_angles'])
    dataset['cluster_aoa_powers'] = np.asarray(dataset['cluster_aoa_powers'])

end_time = time.perf_counter()
elapsed_time_music = end_time - start_time
print(f"Total Execution Time: {elapsed_time_music:.2f} seconds\n\n")


# --- 3. Save Intermediate Results ---
for dataset in all_datasets:
    dataset_name = os.path.basename(dataset['filename'])
    np.save(os.path.join("aoa_estimates_PCA", dataset_name + ".aoa_angles.npy"), np.asarray(dataset["cluster_aoa_angles"]))
    np.save(os.path.join("aoa_estimates_PCA", dataset_name + ".aoa_powers.npy"), np.asarray(dataset["cluster_aoa_powers"]))

# --- 4. Evaluation and Results Summary ---
# Create the directory for the summary file.
plots_output_dir = "plots_3_AoA_Estimation_RAW_PCA" 
round_plots_dir = os.path.join(plots_output_dir, f"Round_{round_num}")
os.makedirs(plots_output_dir, exist_ok=True)
os.makedirs(round_plots_dir, exist_ok=True)

# This list will hold all the lines of text for the final output file.
mae_results_lines = []
mae_results_lines.append(f"Total Execution Time: {elapsed_time_music:.2f} seconds\n\n")

# Add PCA statistics summary to the log file
mae_results_lines.append("--- PCA Statistics Summary ---\n")

if n_components_real_list:
    mae_results_lines.append("\n[REAL Part Components]\n")
    mae_results_lines.append(f"  - Average components used (n_components_): {np.mean(n_components_real_list):.2f}\n")
    mae_results_lines.append(f"  - Max components used: {np.max(n_components_real_list)}\n")
    mae_results_lines.append(f"  - Min components used: {np.min(n_components_real_list)}\n")
    mae_results_lines.append(f"  - Average energy preserved (explained_variance_ratio_): {np.mean(energy_ratio_real_list)*100:.2f}%\n")
    if singular_values_real_list:
        mae_results_lines.append(f"  - Average energy magnitude (singular_values_): {np.mean(singular_values_real_list):.4f}\n")
        mae_results_lines.append(f"  - Max energy magnitude: {np.max(singular_values_real_list):.4f}\n")
        mae_results_lines.append(f"  - Min energy magnitude: {np.min(singular_values_real_list):.4f}\n")

if n_components_imag_list:
    mae_results_lines.append("\n[IMAGINARY Part Components]\n")
    mae_results_lines.append(f"  - Average components used (n_components_): {np.mean(n_components_imag_list):.2f}\n")
    mae_results_lines.append(f"  - Max components used: {np.max(n_components_imag_list)}\n")
    mae_results_lines.append(f"  - Min components used: {np.min(n_components_imag_list)}\n")
    mae_results_lines.append(f"  - Average energy preserved (explained_variance_ratio_): {np.mean(energy_ratio_imag_list)*100:.2f}%\n")
    if singular_values_imag_list:
        mae_results_lines.append(f"  - Average energy magnitude (singular_values_): {np.mean(singular_values_imag_list):.4f}\n")
        mae_results_lines.append(f"  - Max energy magnitude: {np.max(singular_values_imag_list):.4f}\n")
        mae_results_lines.append(f"  - Min energy magnitude: {np.min(singular_values_imag_list):.4f}\n")

mae_results_lines.append("--- Mean Absolute Error (MAE) Summary ---\n\n")

# Loop through only the test datasets to calculate MAE.
for dataset in tqdm(test_set_robot + test_set_human, desc="Calculating MAE for Summary"):
    
    # Add the dataset filename to our summary list.
    dataset_name = os.path.basename(dataset['filename'])
    mae_results_lines.append(f"Dataset: {dataset_name}\n")
    
    # Calculate ideal AoAs (ground truth) for comparison.
    relative_pos = dataset['cluster_positions'][:,np.newaxis,:] - espargos_0007.array_positions
    normal = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_normalvectors)
    right = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_rightvectors)
    ideal_aoas = np.arctan2(right, normal)
    
    # Calculate the estimation errors.
    estimation_errors = dataset['cluster_aoa_angles'] - ideal_aoas
    
    # Loop through each of the 4 arrays to calculate and record its MAE.
    for b in range(estimation_errors.shape[1]):
        # Filter out NaN values that might result from failed estimations.
        valid_errors = estimation_errors[:,b][~np.isnan(estimation_errors[:,b])]
        
        # Calculate MAE in degrees if there are valid error values.
        if valid_errors.size > 0:
            mae = np.mean(np.abs(np.rad2deg(valid_errors)))
            mae_results_lines.append(f"  - Array {b}: MAE = {mae:.4f}°\n")
        else:
            # If all estimates for an array failed, record it as NaN.
            mae_results_lines.append(f"  - Array {b}: MAE = NaN\n")
    
    # Add a blank line between datasets for readability.
    mae_results_lines.append("\n")

# Define the path for the output summary file.
output_txt_path = os.path.join(round_plots_dir, "raw_pca_mae_summary.txt")

# Write all the collected lines to the summary file.
with open(output_txt_path, 'w') as f:
    f.writelines(mae_results_lines)

print(f"RAW data with PCA MAE summary saved to: {os.path.abspath(output_txt_path)}")