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

        # Calculate the spatial covariance matrix R for each of the 4 receiver arrays
        R_old = np.zeros((espargos_0007.ARRAY_COUNT, espargos_0007.COL_COUNT, espargos_0007.COL_COUNT), dtype = np.float32)
        R = np.zeros((espargos_0007.ARRAY_COUNT, espargos_0007.COL_COUNT, espargos_0007.COL_COUNT), dtype = np.float32)

        for tx_csi in csi_by_transmitter_noclutter:
            # Average the covariance matrices from all transmitters
            #print("Sinal1" + str(tx_csi.shape))
            #print("Sinal2", tx_csi[0, 0, 0, 0, 0])
            R_old = R_old + np.float32(np.einsum("dbrms,dbrns->bmn", tx_csi, np.conj(tx_csi)) / tx_csi.shape[0])
            # Snapshot, array, linha, coluna (antenas), subportadora
            for snapshot_i in range(tx_csi.shape[0]):
                for array_i in range(tx_csi.shape[1]):
                    # Fazer o reshape a partir daqui para esses 4 vetores (tem que considerar que tem que fazer a matriz de correlação)
                    # Com o vetor grande aplicar a PCA.transform, selecionar os componentes de maior energia, aplicar PCA.transform_inverse
                    for linha_i in range(tx_csi.shape[2]):
                        for antena_i in range(tx_csi.shape[3]):
                            for antena_j in range(tx_csi.shape[3]):
                                for subportadora_i in range(tx_csi.shape[4]):
                                    sinal_bruto_i = tx_csi[snapshot_i, array_i, linha_i, antena_i, subportadora_i] # Criar um vetor por array (4 vetores) a partir daqui para aplicação da PCA/KPCA a nível de antenas
                                    sinal_bruto_j = tx_csi[snapshot_i, array_i, linha_i, antena_j, subportadora_i]
                                    R[array_i, antena_i, antena_j] = R[array_i, antena_i, antena_j] + np.float32(sinal_bruto_i * np.conj(sinal_bruto_j) / tx_csi.shape[0])

            print("R_old - R: ", np.sum(R_old - R))