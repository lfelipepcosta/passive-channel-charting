from tqdm.auto import tqdm
import espargos_0007
import cluster_utils
import numpy as np
import CRAP
import os
import time
import sys

import SPICE

# Get the round number from the command-line arguments
round_num = sys.argv[1] if len(sys.argv) > 1 else '1'

print("--- Running AoA Estimation with SPICE Implementation ---")

# --- 1. Data Loading and Preprocessing ---
training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)
all_datasets = training_set_robot + test_set_robot + test_set_human

for dataset in all_datasets:
    dataset['clutter_acquisitions'] = np.load(os.path.join("clutter_channel_estimates", os.path.basename(dataset['filename']) + ".npy"))

os.makedirs("aoa_estimates_SPICE", exist_ok=True)

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
        
        # Define physical parameters required by SPICE
        num_antennas_per_array = espargos_0007.COL_COUNT
        num_sources = 1 # We assume a single target
        SPEED_OF_LIGHT = 3e8
        
        # Use the more rigorous wideband-aware calculation for antenna spacing
        CARRIER_FREQUENCY_HZ = 2.472e9  # 2.472 GHz, for Wi-Fi Channel 13
        BANDWIDTH_HZ = 16.56e6          # ~16.56 MHz
        
        # 1. Calculate the maximum operating frequency (f_max)
        max_frequency = CARRIER_FREQUENCY_HZ + (BANDWIDTH_HZ / 2)
        
        # 2. Calculate the minimum wavelength (λ_min) corresponding to f_max
        min_wavelength = SPEED_OF_LIGHT / max_frequency

        # 3. Define the physical spacing 'd' based on the half-wavelength rule
        ANTENNA_SPACING_METERS = min_wavelength / 2.0

        # The center frequency is still used inside the steering vector calculation,
        # which is standard practice. The important part is that 'd' is defined correctly.
        CENTER_FREQUENCY_HZ = 2.472e9
        
        spice_angles_for_cluster = []
        spice_powers_for_cluster = []

        for array_idx in range(R.shape[0]):
            covariance_matrix_for_array = R[array_idx]
            print("Shape da matriz de cov: "+ str(covariance_matrix_for_array.shape))
            
            try:
                # --- This is where we call our SPICE function ---
                angles_rad, powers_db = SPICE.estimate_spice_from_R(
                    covariance_matrix_for_array, 
                    num_antennas_per_array, 
                    ANTENNA_SPACING_METERS, 
                    SPEED_OF_LIGHT, 
                    CENTER_FREQUENCY_HZ, 
                    num_sources
                )
                
                if angles_rad.size > 0:
                    spice_angles_for_cluster.append(angles_rad[0])
                    spice_powers_for_cluster.append(powers_db[0])
                else:
                    spice_angles_for_cluster.append(np.nan)
                    spice_powers_for_cluster.append(np.nan)

            except Exception as e:
                print(f"Warning: SPICE failed for a cluster. Error: {e}")
                spice_angles_for_cluster.append(np.nan)
                spice_powers_for_cluster.append(np.nan)

        dataset['cluster_aoa_angles'].append(np.asarray(spice_angles_for_cluster))
        dataset['cluster_aoa_powers'].append(np.asarray(spice_powers_for_cluster))

    dataset['cluster_aoa_angles'] = np.asarray(dataset['cluster_aoa_angles'])
    dataset['cluster_aoa_powers'] = np.asarray(dataset['cluster_aoa_powers'])

end_time = time.perf_counter()
elapsed_time_spice = end_time - start_time
print(f"\n--- Total Execution Time (SPICE): {elapsed_time_spice:.2f} seconds ---\n")

# --- 3. Save Intermediate Results ---
for dataset in all_datasets:
    dataset_name = os.path.basename(dataset['filename'])
    np.save(os.path.join("aoa_estimates_SPICE", dataset_name + ".aoa_angles.npy"), np.asarray(dataset["cluster_aoa_angles"]))
    np.save(os.path.join("aoa_estimates_SPICE", dataset_name + ".aoa_powers.npy"), np.asarray(dataset["cluster_aoa_powers"]))

# --- 4. Evaluation and Results Summary ---
# Create the directory for the summary file.
plots_output_dir = "plots_3_AoA_Estimation_SPICE" 
round_plots_dir = os.path.join(plots_output_dir, f"Round_{round_num}")
os.makedirs(round_plots_dir, exist_ok=True)

# This list will hold all the lines of text for the final output file.
mae_results_lines = []
mae_results_lines.append(f"Total Execution Time: {elapsed_time_spice:.2f} seconds\n\n")

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
output_txt_path = os.path.join(round_plots_dir, "spice_mae_summary.txt")

# Write all the collected lines to the summary file.
with open(output_txt_path, 'w') as f:
    f.writelines(mae_results_lines)

print(f"SPICE MAE summary saved to: {os.path.abspath(output_txt_path)}")