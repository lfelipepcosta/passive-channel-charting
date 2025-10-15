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

# Group the data into temporal clusters
for dataset in all_datasets:
    cluster_utils.cluster_dataset(dataset)

for dataset in tqdm(all_datasets):
    print(f"AoA estimation for dataset: {dataset['filename']}")

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
        R_old = np.zeros((espargos_0007.ARRAY_COUNT, espargos_0007.COL_COUNT, espargos_0007.COL_COUNT), dtype = np.float32)
        R_array = np.zeros((espargos_0007.ARRAY_COUNT, espargos_0007.COL_COUNT, espargos_0007.COL_COUNT), dtype = np.float32)

        for tx_csi in csi_by_transmitter_noclutter:

            # tx_csi é um tensor do NumPy com 5 dimensões (snapshot, array, linha, coluna, subportadora) 

            if DEBUG:
                print(f"[DEBUG] Shape of 'tx_csi' tensor: {tx_csi.shape}")
                print(f"[DEBUG] Value of one signal element: {tx_csi[0, 0, 0, 0, 0]}")

            # Average the covariance matrices from all transmitters
            R_old = R_old + np.float32(np.einsum("dbrms,dbrns->bmn", tx_csi, np.conj(tx_csi)) / tx_csi.shape[0])

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

                    # Aplicar pré processamento em cima do vetor por array após o reshape

                    # PLACEHOLDER
                    processed_vector_for_single_array = reshaped_vector_for_single_array  # .shape = (53, 8)

                    temp_reshaped = processed_vector_for_single_array.reshape(num_subcarriers, num_rows, num_antennas_per_array)  # .shape = (53, 2, 4)

                    processed_csi_for_single_array = np.transpose(temp_reshaped, (1, 2, 0)) # .shape = (2, 4, 53)
                    
                    if DEBUG:
                        print(f"[DEBUG] ARRAY! Snapshot {snapshot_i}, Array {array_i}:")
                        print(f"[DEBUG]   - Shape before reshaping: {csi_for_single_array_snapshot.shape}")
                        print(f"[DEBUG]   - Shape after reshaping:  {reshaped_vector_for_single_array.shape}")
                        print(f"[DEBUG]   - Shape after reshaping back:   {processed_csi_for_single_array.shape}\n")

                    for row_i in range(num_rows):
                        for antenna_i in range(num_antennas_per_array):
                            for antenna_j in range(num_antennas_per_array):
                                for subcarrier_i in range(num_subcarriers):

                                    single_array_raw_signal_i = processed_csi_for_single_array[row_i, antenna_i, subcarrier_i] # Criar um vetor por array (4 vetores) a partir daqui para aplicação da PCA/KPCA a nível de antenas
                                    single_array_raw_signal_j = processed_csi_for_single_array[row_i, antenna_j, subcarrier_i]
                                    R_array[array_i, antenna_i, antenna_j] = R_array[array_i, antenna_i, antenna_j] + np.float32(single_array_raw_signal_i * np.conj(single_array_raw_signal_j) / num_snapshots)
           
            print_difference = True
            if print_difference:
                difference_array = np.sum(R_old - R_array)
                print(f"[DEBUG] Sum of the difference between R_old and R_array: {difference_array}\n")