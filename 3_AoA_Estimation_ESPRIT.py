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

# Importa o nosso novo módulo com a implementação do ESPRIT
import ESPRIT

print("--- Running AoA Estimation with ESPRIT Implementation ---")

# Data Loading and Preprocessing (igual ao seu script original)
training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)

all_datasets = training_set_robot + test_set_robot + test_set_human

for dataset in all_datasets:
    dataset['clutter_acquisitions'] = np.load(os.path.join("clutter_channel_estimates", os.path.basename(dataset['filename']) + ".npy"))

os.makedirs("aoa_estimates", exist_ok=True)

for dataset in all_datasets:
    cluster_utils.cluster_dataset(dataset)

# Main AoA Estimation Loop
start_time = time.perf_counter()
for dataset in tqdm(all_datasets):
    print(f"AoA estimation for dataset: {dataset['filename']}")

    dataset['cluster_aoa_angles'] = []
    dataset['cluster_aoa_powers'] = []

    for cluster in tqdm(dataset['clusters']):
        csi_by_transmitter_noclutter = []
        for tx_idx, csi in enumerate(cluster['csi_freq_domain']):
            csi_by_transmitter_noclutter.append(CRAP.remove_clutter(csi, dataset['clutter_acquisitions'][tx_idx]))

        R = np.zeros((espargos_0007.ARRAY_COUNT, espargos_0007.COL_COUNT, espargos_0007.COL_COUNT), dtype = np.complex64)

        for tx_csi in csi_by_transmitter_noclutter:
            R = R + np.einsum("dbrms,dbrns->bmn", tx_csi, np.conj(tx_csi)) / tx_csi.shape[0]
        
        # Parameters for our ESPRIT implementation
        num_antennas_per_array = espargos_0007.COL_COUNT  # This is 4
        num_sources = 1                                   # We assume a single target

        # Physical constants and parameters from the paper
        SPEED_OF_LIGHT = 3e8  # m/s
        CARRIER_FREQUENCY_HZ = 2.472e9  # 2.472 GHz, for Wi-Fi Channel 13
        BANDWIDTH_HZ = 16.56e6          # ~16.56 MHz

        # 1. Calculate the maximum operating frequency (f_max) of the signal
        max_frequency = CARRIER_FREQUENCY_HZ + (BANDWIDTH_HZ / 2)

        # 2. Calculate the minimum wavelength (λ_min) corresponding to f_max
        min_wavelength = SPEED_OF_LIGHT / max_frequency

        # 3. Define the physical spacing 'd' based on the half-wavelength rule (Nyquist criterion)
        #    This is the most accurate value for 'd' based on the system parameters.
        ANTENNA_SPACING_METERS = min_wavelength / 2.0

        # 4. Calculate the wavelength (λ_c) for the carrier frequency, used for normalization
        carrier_wavelength = SPEED_OF_LIGHT / CARRIER_FREQUENCY_HZ

        # 5. Calculate the final normalized spacing (d/λ) to be used in the ESPRIT algorithm
        normalized_spacing = ANTENNA_SPACING_METERS / carrier_wavelength
        
        esprit_angles_for_cluster = []
        esprit_powers_for_cluster = [] # ESPRIT doesn't naturally provide a power/confidence metric like Root-MUSIC.
                                       # We will use a placeholder value of 1.0.

        # Iterate over each of the 4 receiver arrays
        for array_idx in range(R.shape[0]):
            covariance_matrix_for_array = R[array_idx]
            
            # Call the ESPRIT function
            # It returns an array of angles; we take the first element since num_sources=1
            try:
                angle_rad = ESPRIT.esprit_implementation(covariance_matrix_for_array, num_antennas_per_array, num_sources, normalized_spacing)[0]
                esprit_angles_for_cluster.append(angle_rad)
                esprit_powers_for_cluster.append(1.0) # Placeholder for power
            except Exception as e:
                # If ESPRIT fails for any reason (e.g., matrix is singular), append NaN
                print(f"Warning: ESPRIT failed for a cluster. Error: {e}")
                esprit_angles_for_cluster.append(np.nan)
                esprit_powers_for_cluster.append(np.nan)

        # Append the results for this cluster
        dataset['cluster_aoa_angles'].append(np.asarray(esprit_angles_for_cluster))
        dataset['cluster_aoa_powers'].append(np.asarray(esprit_powers_for_cluster))
        
        # ===================================================================
        # FIM DA LÓGICA DE SUBSTITUIÇÃO
        # ===================================================================

    dataset['cluster_aoa_angles'] = np.asarray(dataset['cluster_aoa_angles'])
    dataset['cluster_aoa_powers'] = np.asarray(dataset['cluster_aoa_powers'])

# O resto do script (salvar resultados e plotar) permanece o mesmo,
# pois ele apenas usa os dados salvos em 'cluster_aoa_angles' e 'cluster_aoa_powers'.

end_time = time.perf_counter()
elapsed_time_esprit = end_time - start_time
print(f"\n--- Tempo de Execução Total (ESPRIT): {elapsed_time_esprit:.2f} segundos ---\n")
for dataset in all_datasets:
    dataset_name = os.path.basename(dataset['filename'])
    np.save(os.path.join("aoa_estimates", dataset_name + ".aoa_angles.npy"), np.asarray(dataset["cluster_aoa_angles"]))
    np.save(os.path.join("aoa_estimates", dataset_name + ".aoa_powers.npy"), np.asarray(dataset["cluster_aoa_powers"]))

# Evaluation and Visualization
plots_output_dir = "plots_3_AoA_Estimation_ESPRIT" 
os.makedirs(plots_output_dir, exist_ok=True)

for dataset in tqdm(test_set_robot + test_set_human):
    # O código de plotagem e avaliação não precisa de nenhuma alteração
    relative_pos = dataset['cluster_positions'][:,np.newaxis,:] - espargos_0007.array_positions
    normal = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_normalvectors)
    right = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_rightvectors)
    up = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_upvectors)
    ideal_aoas = np.arctan2(right, normal)
    dataset['cluster_groundtruth_aoas'] = ideal_aoas
    dataset['cluster_aoa_estimation_errors'] = dataset['cluster_aoa_angles'] - dataset['cluster_groundtruth_aoas']
    
    # Remove NaN values that may have occurred from ESPRIT failures before calculating MAE
    valid_indices = ~np.isnan(dataset['cluster_aoa_estimation_errors'])
    
    norm = mcolors.Normalize(vmin=-45, vmax=45)
    for b in range(dataset["cluster_aoa_angles"].shape[1]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        errors_for_array = dataset['cluster_aoa_estimation_errors'][:,b][valid_indices[:,b]]
        mae = np.mean(np.abs(np.rad2deg(errors_for_array))) if errors_for_array.size > 0 else float('nan')

        im1 = axes[0].scatter(dataset["cluster_positions"][:,0], dataset["cluster_positions"][:,1], c = np.rad2deg(dataset["cluster_aoa_angles"][:,b]), norm = norm)
        axes[0].set_title(f"ESPRIT AoA Estimates from Array {b}")
        # ... (resto do código de plotagem)
        im2 = axes[1].scatter(dataset["cluster_positions"][:,0], dataset["cluster_positions"][:,1], c = np.rad2deg(dataset["cluster_groundtruth_aoas"][:,b]), norm = norm)
        axes[1].set_title(f"Ideal AoAs from Array {b}")
        # ... (resto do código de plotagem)
        im3 = axes[2].scatter(dataset["cluster_positions"][:,0], dataset["cluster_positions"][:,1], c = np.rad2deg(dataset["cluster_aoa_estimation_errors"][:,b]), norm = norm)
        axes[2].set_title(f"ESPRIT AoA Error from Array {b}, MAE = {mae:.2f}°")
        # ... (resto do código de plotagem)
        
        for ax in axes:
            ax.set_xlabel("x coordinate in m")
            ax.set_ylabel("y coordinate in m")

        cbar_ax = fig.add_axes([1.00, 0.2, 0.02, 0.6])
        fig.colorbar(im1, cax=cbar_ax, label='Angle in Degrees')
        
        plt.tight_layout()
        
        safe_dataset_basename = os.path.basename(dataset['filename']).replace(".tfrecords", "")
        
        plot_filename = f"esprit_aoa_array{b}_{safe_dataset_basename}.png"
        full_plot_path = os.path.join(plots_output_dir, plot_filename)
        
        plt.savefig(full_plot_path, bbox_inches='tight')
        plt.close(fig)
        
print(f"Plots for ESPRIT AoA Estimation saved to: {os.path.abspath(plots_output_dir)}")