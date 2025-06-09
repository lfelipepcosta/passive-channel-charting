# Combine multiple datapoints from different transmitters into clusters of certain duration (e.g., cluster_interval = 1 second).
# Each cluster will contain datapoints from every transmitter.

from tqdm.auto import tqdm
import numpy as np

def cluster_dataset(dataset, cluster_interval = 1):
    """
    Groups individual CSI datapoints into temporal clusters of a fixed duration.
    This function modifies the input 'dataset' dictionary in-place.
    """
    
    # Initialize new keys in the dataset dictionary to store cluster results
    dataset["clusters"] = []
    dataset["cluster_positions"] = []
    dataset["cluster_timestamps"] = []

    print(f"Clustering dataset {dataset['filename']}")
    current_cluster = None

    def finish_cluster():
        """Helper function to finalize, validate, and store a completed cluster."""

        nonlocal current_cluster

        # Calculate the cluster's mean position and final timestamp
        current_cluster["last_timestamp"] = current_cluster["timestamps"][-1]
        current_cluster["mean_position"] = np.mean(current_cluster["groundtruth_positions"], axis = 0)
        # Convert the lists of CSI measurements into NumPy arrays for each transmitter
        for mac in current_cluster["csi_freq_domain"].keys():
            current_cluster["csi_freq_domain"][mac] = np.asarray(current_cluster["csi_freq_domain"][mac])

        # Replace the dict of CSI data with a simple list of the arrays
        current_cluster["csi_freq_domain"] = [csi for csi in current_cluster["csi_freq_domain"].values()]

        # Check if the cluster has received data from all transmitters
        datapoint_count = np.asarray([csi.shape[0] for csi in current_cluster["csi_freq_domain"]])
        if np.any(datapoint_count == 0):
            # If any transmitter is missing, discard this cluster
            print("Warning: Cluster has missing TX, ignoring:", datapoint_count)
        else:
            # If valid, append the cluster and its metadata to the results lists
            dataset["cluster_positions"].append(current_cluster["mean_position"])
            dataset["cluster_timestamps"].append(np.mean(current_cluster["timestamps"]))
            dataset["clusters"].append(current_cluster)
        # Reset the current cluster to start a new one
        current_cluster = None
    
    # Iterate through every single datapoint (ordered by time)
    for d in tqdm(range(dataset["timestamps"].shape[0])):
        # If a cluster is being built, check if its time window has ended
        if current_cluster is not None:
            if dataset["timestamps"][d] > current_cluster["first_timestamp"] + cluster_interval:
                finish_cluster()
                
        # If there is no active cluster, start a new one
        if current_cluster is None:
            current_cluster = dict()
            current_cluster["first_timestamp"] = dataset["timestamps"][d]
            current_cluster["groundtruth_positions"] = []
            current_cluster["timestamps"] = []
            current_cluster["csi_freq_domain"] = dict()
            for mac in dataset["unique_macs"]:
                current_cluster["csi_freq_domain"][mac] = []

        # On every iteration: Add CSI, timestamp and position to cluster
        # Accumulate the current datapoint's information into the active cluster
        mac = dataset["source_macs"][d]
        current_cluster["csi_freq_domain"][mac].append(dataset["csi_freq_domain"][d])
        current_cluster["groundtruth_positions"].append(dataset["groundtruth_positions"][d])
        current_cluster["timestamps"].append(dataset["timestamps"][d])

    # After the loop, finalize the very last cluster if it exists
    if current_cluster is not None:
        finish_cluster()

    # Convert the final lists of positions and timestamps to NumPy arrays
    dataset["cluster_positions"] = np.asarray(dataset["cluster_positions"])
    dataset["cluster_timestamps"] = np.asarray(dataset["cluster_timestamps"])