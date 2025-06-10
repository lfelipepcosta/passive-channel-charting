import FeatureEngineering
import tensorflow as tf
import espargos_0007
import cluster_utils
import neural_network_utils
import CCEvaluation
import numpy as np
import os
import multiprocessing
import matplotlib
matplotlib.use('Agg')       # Use a non-interactive backend for saving plots to files


def combine_datasets(datasets):
    """
    Concatenates features and positions from a list of dataset dictionaries.
    """
    all_cluster_features = []
    all_cluster_positions = []

    # Collect features and 2D positions from each dataset.
    for dataset_idx, dataset in enumerate(datasets):
        all_cluster_features.append(dataset["cluster_features"])
        all_cluster_positions.append(dataset["cluster_positions"][:,:2]) 

    # Combine all lists into single large NumPy arrays
    final_features = np.concatenate(all_cluster_features)
    final_positions = np.concatenate(all_cluster_positions)
    
    return final_positions, final_features


def to_supervised_dataset(datasets):
    """
    Converts a list of dataset dictionaries into a single, combined tf.data.Dataset by creating a dataset for each and concatenating them in a loop.
    """
    # Create an initial dataset from the first file's data
    supervised_dataset = tf.data.Dataset.from_tensor_slices((datasets[0]["cluster_features"], datasets[0]["cluster_positions"][:,:2]))
    # Loop through the rest of the datasets and concatenate them
    for dataset in datasets[1:]:
        supervised_dataset = supervised_dataset.concatenate(tf.data.Dataset.from_tensor_slices((dataset["cluster_features"], dataset["cluster_positions"][:,:2])))

    return supervised_dataset


def train_model(training_features, training_labels):
    """
    Constructs, compiles, and trains the supervised neural network model.
    """
    # Training hyperparameters
    TRAINING_BATCHES = 4000
    BATCH_SIZES = [64, 128, 256, 512, 1024, 2048, 4096]

    # Construct the base NN model architecture
    supervised_model = neural_network_utils.construct_model(input_shape = FeatureEngineering.FEATURE_SHAPE, name = "SupervisedBaselineModel")

    # Build a wrapper model for efficient training using FeatureProviderLayer
    training_input = tf.keras.layers.Input(shape = (), dtype = tf.int64)
    featprov = neural_network_utils.FeatureProviderLayer(dtype = tf.int64)
    featprov.set_features(training_features)
    csi_layer = featprov(training_input)
    output = supervised_model(csi_layer)
    training_model = tf.keras.models.Model(training_input, output, name = "TrainingModel")

    # Compile the model with an optimizer and a loss function for regression
    training_model.compile(optimizer = tf.keras.optimizers.Adam(), loss = tf.keras.losses.MeanSquaredError())

    # Define a Python generator to create random batches of indices and labels
    def random_index_batch_generator():
        batch_count = 0
        while True:
            # Dynamically increase batch size as training progresses
            current_batch_size_index = min(len(BATCH_SIZES) - 1, int(np.floor(batch_count / (TRAINING_BATCHES + 1) * len(BATCH_SIZES))))
            batch_size = BATCH_SIZES[current_batch_size_index]
            batch_count = batch_count + 1
            # Generate random indices and get the corresponding position labels
            indices = np.random.randint(training_features.shape[0], size = batch_size)
            positions = training_labels[indices,:2]
            yield indices.astype(np.int64), positions.astype(np.float32)

    # Convert the Python generator into a tf.data.Dataset
    training_dataset = tf.data.Dataset.from_generator(random_index_batch_generator,
        output_signature=(tf.TensorSpec(shape=(None,), dtype=tf.int64),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32)))

    # Train the model
    training_model.fit(training_dataset, steps_per_epoch = TRAINING_BATCHES)

    # Return the trained inner model (without the FeatureProviderLayer wrapper)
    return supervised_model


if __name__ == '__main__':
    # Necessary for multiprocessing to work correctly on Windows/macOS
    multiprocessing.freeze_support()

    # Create directory for output plots
    plots_output_dir = "plots_2_SupervisedBaseline" 
    os.makedirs(plots_output_dir, exist_ok=True)

    # Load All Datasets
    training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
    test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
    test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)

    all_datasets = training_set_robot + test_set_robot + test_set_human

    # ull Preprocessing Pipeline with Validations
    for dataset_idx, dataset in enumerate(all_datasets):
        # Load the pre-computed clutter subspaces
        clutter_file_path = os.path.join("clutter_channel_estimates", os.path.basename(dataset["filename"]) + ".npy")
        if os.path.exists(clutter_file_path):
            dataset["clutter_acquisitions"] = np.load(clutter_file_path)
        else:
            print(f"ERROR: Clutter file not found: {clutter_file_path}.")


    for dataset_idx, dataset in enumerate(all_datasets):
        # Group data into temporal clusters if clutter data is available
        if "clutter_acquisitions" not in dataset:
            print(f"Error: Dataset {dataset['filename']} is missing 'clutter_acquisitions'. Skipping clustering.")
            continue
        cluster_utils.cluster_dataset(dataset)

    # Filter for datasets that are valid for feature engineering.
    valid_datasets_for_feature_engineering = []
    for ds_idx, ds in enumerate(all_datasets):
        has_clutter = "clutter_acquisitions" in ds
        has_clusters = "clusters" in ds
        if has_clutter and has_clusters:
            valid_datasets_for_feature_engineering.append(ds)
        else:
            print(f"Skipping feature engineering for {ds.get('filename', 'N/A')} due to missing data (clutter_acquisitions: {has_clutter}, clusters: {has_clusters}).")

    if valid_datasets_for_feature_engineering:
        # Perform the computationally intensive feature engineering step
        FeatureEngineering.precompute_features(valid_datasets_for_feature_engineering)
    else:
        print("ERROR: No valid datasets found for feature engineering. Exiting.")
        exit()

    # Prepare Final Data for Training and Evaluation
    valid_training_robot_sets = [ds for ds in training_set_robot if "cluster_features" in ds]
    training_set_robot_groundtruth_positions, training_set_robot_features = combine_datasets(valid_training_robot_sets)
    
    valid_test_robot_sets = [ds for ds in test_set_robot if "cluster_features" in ds]
    test_set_robot_groundtruth_positions, test_set_robot_features = combine_datasets(valid_test_robot_sets)
    
    valid_test_human_sets = [ds for ds in test_set_human if "cluster_features" in ds]
    test_set_human_groundtruth_positions, test_set_human_features = combine_datasets(valid_test_human_sets)
    
    # Create the TensorFlow datasets for evaluation
    training_set_robot_supervised = to_supervised_dataset(valid_training_robot_sets) 
    test_set_robot_supervised = to_supervised_dataset(valid_test_robot_sets)
    test_set_human_supervised = to_supervised_dataset(valid_test_human_sets)

    # Train the Supervised Model
    if training_set_robot_features.shape[0] == 0:
        print("ERROR: Training set is empty. Cannot train model.")
        exit()
    print("Starting model training...")
    supervised_model = train_model(training_set_robot_features, training_set_robot_groundtruth_positions)
    if supervised_model is None:
        print("ERROR: Model training failed. Exiting.")
        exit()
    print("Model training finished.")

    # Evaluate the Model and Generate Results
    print("Evaluating on Training Set (Robot)...")
    # Check if the dataset object for evaluation is valid
    if not isinstance(training_set_robot_supervised, tf.data.Dataset) or training_set_robot_supervised.cardinality().numpy() == 0:
        print("ERROR: training_set_robot_supervised está vazio ou não é um Dataset TF válido. Pulando avaliação no conjunto de treino.")
    else:
        # Get model predictions for the dataset
        training_set_robot_predictions = supervised_model.predict(training_set_robot_supervised.batch(1024))
        # Check if there are groundtruth positions to compare against
        if training_set_robot_groundtruth_positions.shape[0] > 0:
            # Calculate basic metrics for plot titles
            errorvectors, errors, mae, cep = CCEvaluation.compute_localization_metrics(training_set_robot_predictions, training_set_robot_groundtruth_positions)
            suptitle = f"Evaluated on Training Set (Robot)"
            title = f"Supervised: MAE = {mae:.3f}m, CEP = {cep:.3f}m"
            # Generate and save the colorized scatter plot
            plot_filename_train_robot = f"supervised_train_robot_scatter_mae{mae:.3f}_cep{cep:.3f}.png"
            full_plot_path_train_robot = os.path.join(plots_output_dir, plot_filename_train_robot)
            CCEvaluation.plot_colorized(training_set_robot_predictions, training_set_robot_groundtruth_positions, 
                                        suptitle=suptitle, title=title, 
                                        show=False, outfile=full_plot_path_train_robot)
        else:
            print("WARN: training_set_robot_groundtruth_positions está vazio. Não é possível calcular métricas de localização.")


    print("Evaluating on Test Set (Robot)...")
    if not isinstance(test_set_robot_supervised, tf.data.Dataset) or test_set_robot_supervised.cardinality().numpy() == 0:
        print("ERROR: test_set_robot_supervised está vazio ou não é um Dataset TF válido. Pulando avaliação no conjunto de teste robot.")
    else:
        test_set_robot_predictions = supervised_model.predict(test_set_robot_supervised.batch(1024))
        # Calculate basic metrics for plot titles.
        if test_set_robot_groundtruth_positions.shape[0] > 0:
            errorvectors, errors, mae, cep = CCEvaluation.compute_localization_metrics(test_set_robot_predictions, test_set_robot_groundtruth_positions)
            suptitle = f"Evaluated on Test Set (Robot)"
            title = f"Supervised: MAE = {mae:.3f}m, CEP = {cep:.3f}m"
            # Generate and save the colorized scatter plot
            plot_filename_test_robot_scatter = f"supervised_test_robot_scatter_mae{mae:.3f}_cep{cep:.3f}.png"
            full_plot_path_test_robot_scatter = os.path.join(plots_output_dir, plot_filename_test_robot_scatter)
            CCEvaluation.plot_colorized(test_set_robot_predictions, test_set_robot_groundtruth_positions, 
                                        suptitle=suptitle, title=title, 
                                        show=False, outfile=full_plot_path_test_robot_scatter)
            # Calculate the full suite of metrics for the final report
            metrics = CCEvaluation.compute_all_performance_metrics(test_set_robot_predictions, test_set_robot_groundtruth_positions)
            
            ecdf_filename_test_robot = "supervised_test_robot_ecdf.jpg"
            full_ecdf_path_test_robot = os.path.join(plots_output_dir, ecdf_filename_test_robot)
            CCEvaluation.plot_error_ecdf(test_set_robot_predictions, test_set_robot_groundtruth_positions, 
                                          outfile=full_ecdf_path_test_robot)
            # Print the final metrics report
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name.upper().rjust(6, ' ')}: {metric_value:.3f}")
        else:
            print("WARN: test_set_robot_groundtruth_positions está vazio. Não é possível calcular métricas de localização.")


    print("Evaluating on Test Set (Human)...")
    if not isinstance(test_set_human_supervised, tf.data.Dataset) or test_set_human_supervised.cardinality().numpy() == 0:
        print("ERROR: test_set_human_supervised está vazio ou não é um Dataset TF válido. Pulando avaliação no conjunto de teste human.")
    else:
        test_set_human_predictions = supervised_model.predict(test_set_human_supervised.batch(1024))
        if test_set_human_groundtruth_positions.shape[0] > 0:
            errorvectors, errors, mae, cep = CCEvaluation.compute_localization_metrics(test_set_human_predictions, test_set_human_groundtruth_positions)
            suptitle = f"Evaluated on Test Set (Human)"
            title = f"Supervised: MAE = {mae:.3f}m, CEP = {cep:.3f}m"
            plot_filename_test_human_scatter = f"supervised_test_human_scatter_mae{mae:.3f}_cep{cep:.3f}.png"
            full_plot_path_test_human_scatter = os.path.join(plots_output_dir, plot_filename_test_human_scatter)
            CCEvaluation.plot_colorized(test_set_human_predictions, test_set_human_groundtruth_positions, 
                                        suptitle=suptitle, title=title, 
                                        show=False, outfile=full_plot_path_test_human_scatter)
            print(f"Plot salvo em: {full_plot_path_test_human_scatter}")
            
            metrics = CCEvaluation.compute_all_performance_metrics(test_set_human_predictions, test_set_human_groundtruth_positions)
            ecdf_filename_test_human = "supervised_test_human_ecdf.jpg"
            full_ecdf_path_test_human = os.path.join(plots_output_dir, ecdf_filename_test_human)
            CCEvaluation.plot_error_ecdf(test_set_human_predictions, test_set_human_groundtruth_positions, 
                                          outfile=full_ecdf_path_test_human)
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name.upper().rjust(6, ' ')}: {metric_value:.3f}")
        else:
            print("WARN: test_set_human_groundtruth_positions está vazio. Não é possível calcular métricas de localização.")

    print(f"\nPlots for Supervised Baseline saved to: {os.path.abspath(plots_output_dir)}")
    print("Script execution finished.")