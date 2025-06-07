from tqdm.auto import tqdm
import tensorflow as tf
import numpy as np
import hashlib
import json
import os

DATASET_DIR = "espargos-0007"               # Directory where espargos-0007 dataset is stored
MAC_PREFIX = bytes([0x0a, 0xee, 0xf5])      # MAC address prefix to identify the specific Wi-Fi transmitters used in the experiment   

# Lists to store the geometric properties of each antenna array
array_positions = []
array_normalvectors = []
array_upvectors = []
array_rightvectors = []

# Load antenna array specifications from the spec.json file
# This file contains metadata like the physical position and orientation of each ESPARGOS array
with open(os.path.join(DATASET_DIR, "spec.json")) as specfile:
    spec = json.load(specfile)
    for antenna in spec["antennas"]:
        # Append position and orientation vectors for each array
        array_positions.append(np.asarray(antenna["location"]))
        array_upvectors.append(np.asarray(antenna["upvector"]))
        array_rightvectors.append(np.asarray(antenna["rightvector"]))

        # Calculate the normal vector (boresight vector) using the cross product of right and up vectors
        # This vector points in the direction the antenna array is facing
        normalvector = np.cross(np.asarray(antenna["rightvector"]), np.asarray(antenna["upvector"]))
        normalvector = normalvector / np.linalg.norm(normalvector)
        array_normalvectors.append(normalvector)

# Converts lists tu NumPy arrays
array_positions = np.asarray(array_positions)
array_normalvectors = np.asarray(array_normalvectors)
array_upvectors = np.asarray(array_upvectors)
array_rightvectors = np.asarray(array_rightvectors)

centroid = np.mean(array_positions, axis = 0)           # Calculates the geometric center of all antenna arrays

# These constants define the shape of the CSI data, matching the paper's description
ARRAY_COUNT = len(array_positions)      # Number of receiving antennas 
TX_COUNT =  4                           # N_TX = 4
ROW_COUNT = 2                           # M_r = 2
COL_COUNT = 4                           # M_c = 4
SUBCARRIER_COUNT = 53                   # N_ sub = 53

# TFRecord Loading and Preprocessing Helpers

# Describes the structure of each data sample stored in the TFRecord files
# Each feature is defined with its name, type, and shape
feature_description = {
    "csi": tf.io.FixedLenFeature([], tf.string, default_value = ''),        # Channel State Information (raw bytes)
    "time": tf.io.FixedLenFeature([], tf.string, default_value = ''),       # Timestamp (raw bytes)
    "rssi": tf.io.FixedLenFeature([], tf.string, default_value = ''),       # Received Signal Strength Indicator (raw bytes)
    "mac": tf.io.FixedLenFeature([], tf.string, default_value = ''),        # Source MAC address (raw bytes)
    "pos": tf.io.FixedLenFeature([], tf.string, default_value = '')         # Ground truth position (raw bytes)
}

# A function to be mapped over the dataset to parse each raw TFRecord protocol buffer
def record_parse_function(proto):
    # Parse a single binary example into a dictionary of tensors
    record = tf.io.parse_single_example(proto, feature_description)

    # Decode and reshape each tensor to its correct data type and shape
    csi = tf.ensure_shape(tf.io.parse_tensor(record["csi"], out_type = tf.complex64), (ARRAY_COUNT, ROW_COUNT, COL_COUNT, SUBCARRIER_COUNT))
    time = tf.ensure_shape(tf.io.parse_tensor(record["time"], out_type = tf.float64), ())
    rssi = tf.ensure_shape(tf.io.parse_tensor(record["rssi"], out_type = tf.float32), (4, 2, 4))
    pos = tf.ensure_shape(tf.io.parse_tensor(record["pos"], out_type = tf.float64), (3))
    source_mac = tf.ensure_shape(tf.io.parse_tensor(record["mac"], out_type = tf.string), ())

    return csi, pos, time, rssi, source_mac

# A filter function to keep only the data from the known experimental transmitters
def mac_filter(csi, pos, time, rssi, source_mac):
    # Checks if the first 3 bytes of the source MAC address match the defined prefix
    return tf.strings.substr(source_mac, 0, 3) == MAC_PREFIX    # Returns True or False

# A mapping function to weight the CSI by the RSSI
def weight_csi_with_rssi(csi, pos, time, rssi, source_mac):
    # Convert RSSI from dB (logarithmic scale) to linear magnitude
    # Then multiply the CSI by this magnitude. This gives more importance to stronger signals
    csi = tf.cast((10**(rssi / 20))[:,:,:,tf.newaxis], tf.complex64) * csi
    return csi, pos, time, rssi, source_mac

# Main function to load and preprocess a list of TFRecord files
def load_dataset(files):
    datasets = list()           # A list to hold the loaded data from each file

     # Iterate through each filename with a progress bar
    for filename in tqdm(files):
        print(f"Loading {filename}")

        # Create a tf.data pipeline for efficient loading
        # Map applies the function to each element
        tf_dataset = tf.data.TFRecordDataset([filename])        # Create a dataset from the TFRecord file
        tf_dataset = tf_dataset.map(record_parse_function)      # Map the parsing function to decode each record
        tf_dataset = tf_dataset.filter(mac_filter)              # Filter out records that are not from our transmitters
        tf_dataset = tf_dataset.map(weight_csi_with_rssi)       # Map the function to weight CSI by RSSI.

         # Process the pipeline and store data in NumPy arrays
        dataset = dict()
        csi_freq_domain = []
        timestamps = []
        groundtruth_positions = []
        rssis = []
        source_macs = []
        unique_macs = set()
        BATCHSIZE = 1000        # Process the data in batches to manage memory usage

        # Iterate over the dataset in batches
        for csi, pos, time, rssi, source_mac in tf_dataset.batch(BATCHSIZE):
            # Apply ifftshift to the frequency-domain CSI. This is a common step before
            # transforming to the time domain, as it centers the zero-frequency component
            csi_fdomain = np.fft.ifftshift(csi.numpy(), axes = -1)      # Applies a change in the order of frequencies in the last dimension of the array
        
            # Append the NumPy-converted data from the current batch to lists
            csi_freq_domain.append(csi_fdomain)
            timestamps.append(time.numpy())
            rssis.append(rssi.numpy())
            groundtruth_positions.append(pos.numpy())
            for mac in source_mac.numpy():
                macstr = ":".join([f"{b:02x}" for b in mac])    # Converts to hexadecial
                unique_macs.add(macstr)
                source_macs.append(macstr)

         # Concatenate the lists of batch-data into single large NumPy arrays
        csi_freq_domain = np.concatenate(csi_freq_domain)
        timestamps = np.concatenate(timestamps)
        groundtruth_positions = np.concatenate(groundtruth_positions)
        rssis = np.concatenate(rssis)

        # Store all processed data in a dictionary for this file
        dataset["filename"] = filename
        dataset["csi_freq_domain"] = csi_freq_domain
        dataset["timestamps"] = timestamps
        dataset["groundtruth_positions"] = groundtruth_positions
        dataset["rssis"] = rssis
        dataset["source_macs"] = source_macs
        assert(len(unique_macs) == TX_COUNT)            # Verifies if the number of transmitters is right
        dataset["unique_macs"] = sorted(unique_macs)

        datasets.append(dataset)

    return datasets

# Utility Functions and Dataset Definitions

# Creates a unique hash from a list of dataset filenames.
# This is useful for naming cached files of pre-processed data
def hash_dataset_names(datasets):
    filenames = [ds["filename"] for ds in datasets]
    dataset_path_hash = hashlib.sha1(":".join(filenames).encode()).hexdigest()
    return dataset_path_hash

# Files for the training set, using the robot target
TRAINING_SET_ROBOT_FILES = [
    os.path.join(DATASET_DIR, "espargos-0007-meanders-nw-se-1.tfrecords"),
    os.path.join(DATASET_DIR, "espargos-0007-meanders-sw-ne-1.tfrecords"),
    os.path.join(DATASET_DIR, "espargos-0007-randomwalk-1.tfrecords")
]

# Files for the test set, using the robot target
TEST_SET_ROBOT_FILES = [
    os.path.join(DATASET_DIR, "espargos-0007-randomwalk-2.tfrecords")
]

# Files for the test set, using the human target (to test generalization)
TEST_SET_HUMAN_FILES = [
    os.path.join(DATASET_DIR, "espargos-0007-human-helmet-randomwalk-1.tfrecords")
]
