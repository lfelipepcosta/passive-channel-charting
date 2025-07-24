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

import MUSIC

# Pega o número da rodada a partir do argumento da linha de comando
# Se nenhum argumento for passado, assume a rodada 1 como padrão.
round_num = sys.argv[1] if len(sys.argv) > 1 else '1'

print("--- Running AoA Estimation with standard MUSIC Implementation ---")

# --- 1. Data Loading and Preprocessing ---
training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)
all_datasets = training_set_robot + test_set_robot + test_set_human

for dataset in all_datasets:
    dataset['clutter_acquisitions'] = np.load(os.path.join("clutter_channel_estimates", os.path.basename(dataset['filename']) + ".npy"))

os.makedirs("aoa_estimates_MUSIC", exist_ok=True)

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
        
        # Define physical parameters required by MUSIC
        num_antennas_per_array = espargos_0007.COL_COUNT
        num_sources = 1
        SPEED_OF_LIGHT = 3e8
        CENTER_FREQUENCY_HZ = 2.472e9
        wavelength = SPEED_OF_LIGHT / CENTER_FREQUENCY_HZ
        ANTENNA_SPACING_METERS = wavelength / 2.0
        
        music_angles_for_cluster = []
        music_powers_for_cluster = []

        for array_idx in range(R.shape[0]):
            covariance_matrix_for_array = R[array_idx]
            
            try:
                angles_rad, powers_db = MUSIC.estimate_music_from_R(
                    covariance_matrix_for_array, 
                    num_antennas_per_array, 
                    ANTENNA_SPACING_METERS, 
                    SPEED_OF_LIGHT, 
                    CENTER_FREQUENCY_HZ,
                    num_sources
                )
                
                if angles_rad.size > 0:
                    music_angles_for_cluster.append(angles_rad[0])
                    music_powers_for_cluster.append(powers_db[0])
                else:
                    music_angles_for_cluster.append(np.nan)
                    music_powers_for_cluster.append(np.nan)

            except Exception as e:
                print(f"Warning: MUSIC failed for a cluster. Error: {e}")
                music_angles_for_cluster.append(np.nan)
                music_powers_for_cluster.append(np.nan)

        dataset['cluster_aoa_angles'].append(np.asarray(music_angles_for_cluster))
        dataset['cluster_aoa_powers'].append(np.asarray(music_powers_for_cluster))

    dataset['cluster_aoa_angles'] = np.asarray(dataset['cluster_aoa_angles'])
    dataset['cluster_aoa_powers'] = np.asarray(dataset['cluster_aoa_powers'])

end_time = time.perf_counter()
elapsed_time_music = end_time - start_time
print(f"\n--- Total Execution Time (MUSIC): {elapsed_time_music:.2f} seconds ---\n")

# --- 3. Save Results ---
for dataset in all_datasets:
    dataset_name = os.path.basename(dataset['filename'])
    np.save(os.path.join("aoa_estimates_MUSIC", dataset_name + ".aoa_angles.npy"), np.asarray(dataset["cluster_aoa_angles"]))
    np.save(os.path.join("aoa_estimates_MUSIC", dataset_name + ".aoa_powers.npy"), np.asarray(dataset["cluster_aoa_powers"]))

# --- 4. Evaluation and Visualization ---
plots_output_dir = "plots_3_AoA_Estimation_MUSIC" 
round_plots_dir = os.path.join(plots_output_dir, f"Round_{round_num}")
os.makedirs(round_plots_dir, exist_ok=True)
os.makedirs(plots_output_dir, exist_ok=True)

for dataset in tqdm(test_set_robot + test_set_human):
    relative_pos = dataset['cluster_positions'][:,np.newaxis,:] - espargos_0007.array_positions
    normal = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_normalvectors)
    right = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_rightvectors)
    ideal_aoas = np.arctan2(right, normal)
    dataset['cluster_groundtruth_aoas'] = ideal_aoas
    estimation_errors = dataset['cluster_aoa_angles'] - dataset['cluster_groundtruth_aoas']
    
    norm = mcolors.Normalize(vmin=-45, vmax=45)
    for b in range(estimation_errors.shape[1]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        valid_errors = estimation_errors[:,b][~np.isnan(estimation_errors[:,b])]
        mae = np.mean(np.abs(np.rad2deg(valid_errors))) if valid_errors.size > 0 else float('nan')

        axes[0].set_title(f"MUSIC AoA Estimates from Array {b}")
        axes[1].set_title(f"Ideal AoAs from Array {b}")
        axes[2].set_title(f"MUSIC AoA Estimation Error, MAE = {mae:.2f}°")
        
        im1 = axes[0].scatter(dataset["cluster_positions"][:,0], dataset["cluster_positions"][:,1], c = np.rad2deg(dataset["cluster_aoa_angles"][:,b]), norm = norm)
        im2 = axes[1].scatter(dataset["cluster_positions"][:,0], dataset["cluster_positions"][:,1], c = np.rad2deg(dataset["cluster_groundtruth_aoas"][:,b]), norm = norm)
        im3 = axes[2].scatter(dataset["cluster_positions"][:,0], dataset["cluster_positions"][:,1], c = np.rad2deg(estimation_errors[:,b]), norm = norm)
        for ax in axes: ax.set_xlabel("x coordinate in m"); ax.set_ylabel("y coordinate in m")
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]); fig.colorbar(im1, cax=cbar_ax, label="Angle in Degrees")
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        
        safe_dataset_basename = os.path.basename(dataset['filename']).replace(".tfrecords", "")
        plot_filename = f"music_aoa_array{b}_{safe_dataset_basename}.png"
        #plt.savefig(os.path.join(plots_output_dir, plot_filename))
        plt.savefig(os.path.join(round_plots_dir, plot_filename))
        plt.close(fig)
        
print(f"Plots for standard MUSIC AoA Estimation saved to: {os.path.abspath(plots_output_dir)}")