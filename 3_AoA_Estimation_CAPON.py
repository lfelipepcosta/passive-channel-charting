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

from aoa_algorithms import CAPON

# Pega o número da rodada a partir do argumento da linha de comando
# Se nenhum argumento for passado, assume a rodada 1 como padrão.
round_num = sys.argv[1] if len(sys.argv) > 1 else '1'

print("--- Running AoA Estimation with CAPON Implementation ---")

# --- 1. Data Loading and Preprocessing ---
training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)
all_datasets = training_set_robot + test_set_robot + test_set_human

for dataset in all_datasets:
    dataset['clutter_acquisitions'] = np.load(os.path.join("clutter_channel_estimates", os.path.basename(dataset['filename']) + ".npy"))

os.makedirs("aoa_estimates_CAPON", exist_ok=True)

for dataset in all_datasets:
    cluster_utils.cluster_dataset(dataset)

# --- 2. Main AoA Estimation Loop ---
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
        
        # --- Define physical parameters required by Capon ---
        num_antennas_per_array = espargos_0007.COL_COUNT  # This is 4
        num_sources = 1                                   # We assume a single target
        SPEED_OF_LIGHT = 3e8                              # m/s
        CENTER_FREQUENCY_HZ = 2.472e9                     # 2.472 GHz from the paper
        
        # Estimate antenna spacing 'd' as half the wavelength of the center frequency.
        wavelength = SPEED_OF_LIGHT / CENTER_FREQUENCY_HZ
        ANTENNA_SPACING_METERS = wavelength / 2.0
        
        capon_angles_for_cluster = []
        capon_powers_for_cluster = []

        # Iterate over each of the 4 receiver arrays.
        for array_idx in range(R.shape[0]):
            covariance_matrix_for_array = R[array_idx]
            
            # Call the adapted Capon function.
            try:
                angles_rad, powers_db = CAPON.estimate_capon_from_R(
                    covariance_matrix_for_array, 
                    num_antennas_per_array, 
                    ANTENNA_SPACING_METERS, 
                    SPEED_OF_LIGHT, 
                    CENTER_FREQUENCY_HZ, 
                    num_sources
                )
                
                # If Capon finds an angle, take the first one (since num_sources=1).
                if angles_rad.size > 0:
                    capon_angles_for_cluster.append(angles_rad[0])
                    capon_powers_for_cluster.append(powers_db[0]) # Use power as confidence
                else: # If no peak is found, append NaN.
                    capon_angles_for_cluster.append(np.nan)
                    capon_powers_for_cluster.append(np.nan)

            except Exception as e:
                # If Capon fails for any reason, append NaN.
                print(f"Warning: Capon failed for a cluster. Error: {e}")
                capon_angles_for_cluster.append(np.nan)
                capon_powers_for_cluster.append(np.nan)

        # Append the results for this cluster.
        dataset['cluster_aoa_angles'].append(np.asarray(capon_angles_for_cluster))
        dataset['cluster_aoa_powers'].append(np.asarray(capon_powers_for_cluster))

    dataset['cluster_aoa_angles'] = np.asarray(dataset['cluster_aoa_angles'])
    dataset['cluster_aoa_powers'] = np.asarray(dataset['cluster_aoa_powers'])

end_time = time.perf_counter()
elapsed_time_capon = end_time - start_time
print(f"\n--- Total Execution Time (CAPON): {elapsed_time_capon:.2f} seconds ---\n")

# --- 3. Save Results and Plot ---
# The rest of the script remains the same, but we adjust filenames and titles.
for dataset in all_datasets:
    dataset_name = os.path.basename(dataset['filename'])
    # Save results to the same output folder, allowing triangulation to use them.
    np.save(os.path.join("aoa_estimates_CAPON", dataset_name + ".aoa_angles.npy"), np.asarray(dataset["cluster_aoa_angles"]))
    np.save(os.path.join("aoa_estimates_CAPON", dataset_name + ".aoa_powers.npy"), np.asarray(dataset["cluster_aoa_powers"]))

# --- 4. Evaluation and Visualization ---
plots_output_dir = "plots_3_AoA_Estimation_CAPON" 
round_plots_dir = os.path.join(plots_output_dir, f"Round_{round_num}")
os.makedirs(round_plots_dir, exist_ok=True)
os.makedirs(plots_output_dir, exist_ok=True)

for dataset in tqdm(test_set_robot + test_set_human):
    # Calculate ideal AoAs for comparison.
    relative_pos = dataset['cluster_positions'][:,np.newaxis,:] - espargos_0007.array_positions
    normal = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_normalvectors)
    right = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_rightvectors)
    ideal_aoas = np.arctan2(right, normal)
    dataset['cluster_groundtruth_aoas'] = ideal_aoas

    # Calculate the estimation error, ignoring NaNs from failed estimations.
    estimation_errors = dataset['cluster_aoa_angles'] - dataset['cluster_groundtruth_aoas']
    
    # Generate plots for each receiver array.
    norm = mcolors.Normalize(vmin=-45, vmax=45)
    for b in range(estimation_errors.shape[1]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Calculate MAE only for valid (non-NaN) estimates.
        valid_errors = estimation_errors[:,b][~np.isnan(estimation_errors[:,b])]
        mae = np.mean(np.abs(np.rad2deg(valid_errors))) if valid_errors.size > 0 else float('nan')

        # Plot 1: Map colored by the estimated AoA.
        im1 = axes[0].scatter(dataset["cluster_positions"][:,0], dataset["cluster_positions"][:,1], c = np.rad2deg(dataset["cluster_aoa_angles"][:,b]), norm = norm)
        axes[0].set_title(f"Capon AoA Estimates from Array {b}")
        axes[0].set_xlabel("x coordinate in m")
        axes[0].set_ylabel("y coordinate in m")

        # Plot 2: Map colored by the ideal (ground-truth) AoA.
        im2 = axes[1].scatter(dataset["cluster_positions"][:,0], dataset["cluster_positions"][:,1], c = np.rad2deg(dataset["cluster_groundtruth_aoas"][:,b]), norm = norm)
        axes[1].set_title(f"Ideal AoAs from Array {b}")
        axes[1].set_xlabel("x coordinate in m")
        axes[1].set_ylabel("y coordinate in m")

        # Plot 3: Map colored by the estimation error.
        im3 = axes[2].scatter(dataset["cluster_positions"][:,0], dataset["cluster_positions"][:,1], c = np.rad2deg(estimation_errors[:,b]), norm = norm)
        axes[2].set_title(f"Capon AoA Estimation Error, MAE = {mae:.2f}°")
        axes[2].set_xlabel("x coordinate in m")
        axes[2].set_ylabel("y coordinate in m")

        # Add a colorbar to the figure.
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im1, cax=cbar_ax, label="Angle in Degrees")
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        
        # Save the figure to a file.
        safe_dataset_basename = os.path.basename(dataset['filename']).replace(".tfrecords", "")
        plot_filename = f"capon_aoa_array{b}_{safe_dataset_basename}.png"
        #plt.savefig(os.path.join(plots_output_dir, plot_filename))
        plt.savefig(os.path.join(round_plots_dir, plot_filename))
        plt.close(fig)
        
print(f"Plots for Capon AoA Estimation saved to: {os.path.abspath(plots_output_dir)}")