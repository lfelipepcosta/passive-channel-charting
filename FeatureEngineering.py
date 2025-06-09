# Pre-compute features for Neural Networks in NumPy
# Used for both supervised baseline and Channel Charting

#
# This script transforms raw Channel State Information (CSI) into structured features
# suitable for a neural network, as described in Section V of the source paper.
# The process is parallelized for efficiency.
#
# INPUT:
# The input to a worker process is a batch of raw, clutter-cleaned CSI measurements
# from a single time-cluster and a single transmitter. This is the frequency-domain
# channel response (H_tgt) across all antennas and subcarriers for a short time window. 
#
# TRANSFORMATION:
# The goal is to extract a compact and meaningful "fingerprint" from the raw CSI.
# The transformation consists of three main steps:
#
# 1. IFFT (Frequency to Time Domain):
#    The frequency-domain CSI is converted to the time-domain Channel Impulse
#    Response (CIR) using an Inverse Fast Fourier Transform.
#
# 2. Tap Selection:
#    A small window of the most significant time taps (e.g., 12 taps) is
#    extracted from the full CIR. This focuses the analysis on the main signal echo.
#
# 3. Covariance Calculation:
#    For each of the selected time taps and for each receiver array, a spatial
#    covariance matrix is computed. This matrix captures the relationship
#    between all receiving antennas at that specific moment in time.
#
# OUTPUT:
# The output is a high-dimensional feature tensor containing a collection of these
# spatial covariance matrices. This structured tensor explicitly encodes the
# spatio-temporal characteristics of the channel and is much more suitable for
# a neural network to learn from than the raw CSI data.
#

import multiprocessing as mp
from tqdm.auto import tqdm
import espargos_0007
import numpy as np
import CRAP

# Use up to 8 processes, but not more than half the available CPU cores
PROCESSES = min(8, mp.cpu_count() // 2)

TAP_START = espargos_0007.SUBCARRIER_COUNT // 2 - 4
TAP_STOP = espargos_0007.SUBCARRIER_COUNT // 2 + 8
# Helper variable for the flattened antenna dimension size
cov_matrix_shape = espargos_0007.ROW_COUNT * espargos_0007.COL_COUNT
# Define the final shape of the feature tensor for one cluster
FEATURE_SHAPE = (espargos_0007.TX_COUNT, espargos_0007.ARRAY_COUNT, TAP_STOP - TAP_START, cov_matrix_shape, cov_matrix_shape)


def feature_engineering_worker(todo_queue, output_queue):
"""
A worker process that takes a CSI processing task, computes features, and returns the result.
"""

    while True:
        task_details = todo_queue.get()        # Get a task from the queue. This will block until a task is available
        if task_details is None:               # If the task is None, it's a signal to terminate the worker
            break
        
        # Unpack the data required for this specific task
        dataset_idx, cluster_idx, tx_idx, csi_fdomain_data, clutter_acquisitions_data = task_details   
        # Step 1: Remove clutter from the CSI data
        csi_fdomain_noclutter = CRAP.remove_clutter(csi_fdomain_data, clutter_acquisitions_data)            
        # Step 2: Perform IFFT to get the time-domain signal, then extract the relevant taps in one step
        csi_tdomain = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(csi_fdomain_noclutter, axes=-1), axis=-1), axes=-1)[..., TAP_START:TAP_STOP]
        # Step 3: Reshape the data to flatten the per-array antenna dimensions
        csi_tdomain_flat = np.reshape(csi_tdomain, (csi_tdomain.shape[0], espargos_0007.ARRAY_COUNT, cov_matrix_shape, TAP_STOP - TAP_START))
        # Step 4: Compute the time-domain covariance matrices (final features)
        result_data = np.einsum("davt,dawt->atvw", np.conj(csi_tdomain_flat), csi_tdomain_flat)
        # Send the computed features back to the main process via the output queue
        output_queue.put((dataset_idx, cluster_idx, tx_idx, result_data))       


def precompute_features(all_datasets):
"""
Orchestrates the feature engineering process using multiple worker processes.
"""
    # Create queues for communication between the main process and workers
    todo_queue = mp.Queue()
    output_queue = mp.Queue()

    # Setup and start the worker processes
    processes = []
    for _ in range(PROCESSES):

        p = mp.Process(target=feature_engineering_worker, args=(todo_queue, output_queue))
        p.start()
        processes.append(p)

    # Distribute all tasks to the workers
    total_tasks = 0
    # Iterate through each dataset file
    for dataset_idx, dataset in enumerate(all_datasets):
        # Pre-allocate a NumPy array to store the results for this dataset
        dataset["cluster_features"] = np.zeros((len(dataset["clusters"]),) + FEATURE_SHAPE, dtype=np.complex64)
        # Iterate through each cluster in the dataset
        for cluster_idx, cluster in enumerate(dataset["clusters"]):
            # Iterate through each transmitter's data within the cluster
            for tx_idx in range(len(cluster["csi_freq_domain"])):
                total_tasks = total_tasks + 1
                # Package the necessary data for one task into a tuple
                csi_fdomain_task_data = all_datasets[dataset_idx]["clusters"][cluster_idx]["csi_freq_domain"][tx_idx]
                clutter_acquisitions_task_data = all_datasets[dataset_idx]["clutter_acquisitions"][tx_idx]
                task_data_tuple = (dataset_idx, cluster_idx, tx_idx, csi_fdomain_task_data, clutter_acquisitions_task_data)
                # Put the task on the "to-do" queue
                todo_queue.put(task_data_tuple)

    print(f"Pre-computing training features for {total_tasks} datapoints in total")
    
    # Collect results from the workers
    with tqdm(total=total_tasks) as pbar:

        finished_tasks_count = 0
        while finished_tasks_count < total_tasks:

            # Wait for any worker to finish a task and get the result
            dataset_idx_res, cluster_idx_res, tx_idx_res, res_data = output_queue.get()

            # Place the computed feature data in the correct position in the pre-allocated array
            all_datasets[dataset_idx_res]["cluster_features"][cluster_idx_res, tx_idx_res] = res_data
            pbar.update(1)
            finished_tasks_count += 1

    # Shutdown the worker processes cleanly
    # Send a "None" signal for each worker to tell them to exit.
    for _ in range(PROCESSES):
        todo_queue.put(None)

     # Wait for all worker processes to complete their execution
    for p in processes:
        p.join()