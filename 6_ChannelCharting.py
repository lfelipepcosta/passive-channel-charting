import multiprocessing as mp
import os
import espargos_0007
import cluster_utils
import CCEvaluation
import FeatureEngineering
import neural_network_utils
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)

all_datasets = training_set_robot + test_set_robot + test_set_human

for dataset in all_datasets:
    dataset_name = os.path.basename(dataset["filename"])
    dataset["clutter_acquisitions"] = np.load(os.path.join("clutter_channel_estimates", dataset_name + ".npy"))
    if dataset in training_set_robot:
        dataset["cluster_aoa_angles"] = np.load(os.path.join("aoa_estimates", dataset_name + ".aoa_angles.npy"))
        dataset["cluster_aoa_powers"] = np.load(os.path.join("aoa_estimates", dataset_name + ".aoa_powers.npy"))

for dataset in all_datasets:
    cluster_utils.cluster_dataset(dataset)

FeatureEngineering.precompute_features(all_datasets)

def aoa_power_to_kappa(aoa_power):
    return 5 * aoa_power**6

def bessel_i0_approx(x):
	return tf.math.cosh(x) / (1 + x**2 / 4)**(1/4) * (1 + 0.24273 * x**2) / (1 + 0.43023 * x**2)

class ChannelChartingLoss(tf.keras.losses.Loss):
    def __init__(self, classical_weight, aoa_angles, aoa_powers, height, dissimilarity_matrix, dissimilarity_margin = 1, name = "CCLoss"):
        super().__init__(name = name)
        self.classical_weight = classical_weight
        self.height = height
        self.dissimilarity_matrix_tensor = tf.constant(dissimilarity_matrix, dtype = tf.float32)
        self.dissimilarity_margin = dissimilarity_margin
        self.estimated_aoas_tensor = tf.constant(aoa_angles, dtype = tf.float32)
        self.aoa_powers_tensor = tf.constant(aoa_powers, dtype = tf.float32)
        self.centers_tensor = tf.constant(espargos_0007.array_positions, dtype = tf.float32)
        self.normalvectors_tensor = tf.constant(espargos_0007.array_normalvectors, dtype = tf.float32)
        self.rightvectors_tensor = tf.constant(espargos_0007.array_rightvectors, dtype = tf.float32)
        self.centroid_tensor = tf.constant(espargos_0007.centroid, dtype = tf.float32)[tf.newaxis, :2]

    def classical(self, pos, aoas, aoa_powers):
        pos_with_height = tf.concat([pos + self.centroid_tensor, self.height * tf.ones(tf.shape(pos)[0])[:, tf.newaxis]], 1)
        relative_pos = pos_with_height[:,tf.newaxis,:] - self.centers_tensor
        normal = tf.einsum("dax,ax->da", relative_pos, self.normalvectors_tensor)
        right = tf.einsum("dax,ax->da", relative_pos, self.rightvectors_tensor)
        ideal_aoas = tf.math.atan2(right, normal)
        kappas = aoa_power_to_kappa(aoa_powers)
        aoa_likelihoods = tf.exp(kappas * tf.cos(ideal_aoas - aoas)) / (2 * np.pi * bessel_i0_approx(kappas))
        return tf.math.reduce_prod(aoa_likelihoods, axis = -1)

    def siamese(self, pos_A, pos_B, dissimilarities):
        distances_pred = tf.math.sqrt(tf.math.reduce_sum(tf.square(pos_A - pos_B), axis = 1))
        return tf.reduce_mean(tf.square(distances_pred - dissimilarities) / (dissimilarities + self.dissimilarity_margin))

    def call(self, y_true, y_pred):
        index_A = tf.cast(y_true[:,0], tf.int32)
        index_B = tf.cast(y_true[:,1], tf.int32)
        pos_A, pos_B = (y_pred[:,:2], y_pred[:,2:])
        dissimilarities = tf.gather_nd(self.dissimilarity_matrix_tensor, tf.transpose([index_A, index_B]))
        siamese_loss = self.siamese(pos_A, pos_B, dissimilarities)
        aoa_A = tf.gather(self.estimated_aoas_tensor, index_A)
        aoa_B = tf.gather(self.estimated_aoas_tensor, index_B)
        aoa_power_A = tf.gather(self.aoa_powers_tensor, index_A)
        aoa_power_B = tf.gather(self.aoa_powers_tensor, index_B)
        classical_loss = -tf.reduce_sum(self.classical(pos_A, aoa_A, aoa_power_A) + self.classical(pos_B, aoa_B, aoa_power_B))
        return self.classical_weight * classical_loss + (1 - self.classical_weight) * siamese_loss

def combine_datasets(datasets, for_training = False):
    combined = dict()
    combined["cluster_features"] = []
    combined["cluster_positions"] = []
    heights = []
    if for_training:
        combined["cluster_aoa_angles"] = []
        combined["cluster_aoa_powers"] = []

    for dataset in datasets:
        combined["cluster_features"].append(dataset["cluster_features"])
        combined["cluster_positions"].append(dataset["cluster_positions"][:,:2])
        heights.append(dataset["cluster_positions"][:,2])
        if for_training:
            combined["cluster_aoa_angles"].append(dataset["cluster_aoa_angles"])
            combined["cluster_aoa_powers"].append(dataset["cluster_aoa_powers"])

    combined["cluster_features"] = np.concatenate(combined["cluster_features"])
    combined["cluster_positions"] = np.concatenate(combined["cluster_positions"])
    combined["mean_height"] = np.mean(np.concatenate(heights))
    if for_training:
        combined["cluster_aoa_angles"] = np.concatenate(combined["cluster_aoa_angles"])
        combined["cluster_aoa_powers"] = np.concatenate(combined["cluster_aoa_powers"])

    if for_training:
        training_set_name = espargos_0007.hash_dataset_names(datasets)
        combined["dissimilarity_matrix"] = np.load(os.path.join("dissimilarity_matrices", training_set_name + ".geodesic_meters.npy"))

    return combined

def affine_transform_channel_chart(groundtruth_pos, channel_chart_pos):
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:,:-1]
    A, res, rank, s = np.linalg.lstsq(pad(channel_chart_pos), pad(groundtruth_pos), rcond = None)
    transform = lambda x: unpad(np.dot(pad(x), A))
    return transform(channel_chart_pos)

class ChartPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, featprov, position_labels, fcf, augmented = True):
        self.featprov = featprov
        self.position_labels = position_labels
        self.fcf = fcf
        self.augmented = augmented
   
    def on_epoch_end(self, epoch, logs = None):
        position_predictions = self.fcf.predict(self.featprov(np.arange(self.position_labels.shape[0]))) + espargos_0007.centroid[:2]
        suptitle = f"Epoch {epoch}"

        if self.augmented:
            position_predictions_trans = position_predictions
        else:
            suptitle = suptitle + " (after opt. aff. trans.)"
            position_predictions_trans = affine_transform_channel_chart(self.position_labels, position_predictions)

        errorvectors, errors, mae, cep = CCEvaluation.compute_localization_metrics(position_predictions_trans, self.position_labels)
        title = f"Unsupervised: MAE = {mae:.3f}m, CEP = {cep:.3f}m"
        CCEvaluation.plot_colorized(position_predictions_trans, self.position_labels, suptitle = suptitle, title = title, show=False)
        plt.close() # Close plot to prevent showing in GUI and to free memory

def train_model(training_data, augmented = True):
    STEPS_PER_EPOCH = 200
    EPOCHS = 10
    LEARNING_RATE_INITIAL = 1e-2
    LEARNING_RATE_FINAL = 1e-5
    BATCH_SIZES = [64, 128, 256, 512, 1024, 2048, 4096]
    
    training_features = training_data["cluster_features"]
    sample_count = training_features.shape[0]

    fcf_model = neural_network_utils.construct_model(input_shape = FeatureEngineering.FEATURE_SHAPE, name = "ChannelChartingModel")

    input_A = tf.keras.layers.Input(shape = (), dtype = tf.int64)
    input_B = tf.keras.layers.Input(shape = (), dtype = tf.int64)
    featprov = neural_network_utils.FeatureProviderLayer(dtype = tf.int64)
    featprov.set_features(training_features)
    csi_A = featprov(input_A)
    csi_B = featprov(input_B)
    embedding_A = fcf_model(csi_A)
    embedding_B = fcf_model(csi_B)
    output = tf.keras.layers.concatenate([embedding_A, embedding_B], axis = 1)
    siamese_model = tf.keras.models.Model([input_A, input_B], output, name = "SiameseNeuralNetwork")

    loss = ChannelChartingLoss(
        classical_weight = 0.05 if augmented else 0.0,
        aoa_angles = training_data["cluster_aoa_angles"],
        aoa_powers = training_data["cluster_aoa_powers"],
        height = training_data["mean_height"],
        dissimilarity_matrix = training_data["dissimilarity_matrix"],
        dissimilarity_margin = np.quantile(training_data["dissimilarity_matrix"], 0.01),
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate = LEARNING_RATE_INITIAL,
                    decay_steps = EPOCHS * STEPS_PER_EPOCH,
                    decay_rate = LEARNING_RATE_FINAL / LEARNING_RATE_INITIAL,
                    staircase = False)
    
    siamese_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule), loss = loss)

    def random_index_batch_generator():
        batch_count = 0
        while True:
            batch_size = BATCH_SIZES[min(int(np.floor(batch_count / (EPOCHS * STEPS_PER_EPOCH + 1) * len(BATCH_SIZES))), len(BATCH_SIZES) - 1)]
            batch_count = batch_count + 1
            indices_A = np.random.randint(sample_count, size = 256)
            indices_B = np.random.randint(sample_count, size = 256)
            yield (indices_A, indices_B), tf.stack([indices_A, indices_B], axis = 1)

    training_dataset = tf.data.Dataset.from_generator(random_index_batch_generator,
        output_signature = ((tf.TensorSpec(shape = (None,), dtype = tf.int32), tf.TensorSpec(shape = (None,), dtype = tf.int32)),
        tf.TensorSpec(shape = (None, 2), dtype = tf.int32)))

    plot_callback = ChartPlotCallback(featprov, training_data["cluster_positions"], fcf_model, augmented = augmented)
    siamese_model.fit(training_dataset, steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, callbacks = [plot_callback])

    return fcf_model


if __name__ == '__main__':
    mp.freeze_support()
    
    robot_training_data = combine_datasets(training_set_robot, for_training=True)
    robot_test_data = combine_datasets(test_set_robot)
    human_test_data = combine_datasets(test_set_human)
    
    plots_output_dir = "plots_6_ChannelCharting"
    os.makedirs(plots_output_dir, exist_ok=True)

    print("\nUn-augmented model")
    fcf_model = train_model(robot_training_data, augmented=False)
    
    print("\nTest set robot")
    test_set_robot_predictions = fcf_model.predict(robot_test_data["cluster_features"])
    test_set_robot_predictions_transformed = affine_transform_channel_chart(robot_test_data["cluster_positions"], test_set_robot_predictions)
    metrics = CCEvaluation.compute_all_performance_metrics(test_set_robot_predictions_transformed, robot_test_data["cluster_positions"])
    CCEvaluation.plot_colorized(test_set_robot_predictions_transformed, robot_test_data["cluster_positions"], suptitle="PCC on Test Set (Robot)", title=f"Unsupervised: MAE = {metrics['mae']:.3f}m", show=False, outfile=os.path.join(plots_output_dir, "pcc_unaug_robot_scatter.jpg"))
    CCEvaluation.plot_error_ecdf(test_set_robot_predictions_transformed, robot_test_data["cluster_positions"], outfile=os.path.join(plots_output_dir, "pcc_unaug_robot_ecdf.jpg"))
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.upper().rjust(6, ' ')}: {metric_value:.3f}")

    print("\nTest set human")
    test_set_human_predictions = fcf_model.predict(human_test_data["cluster_features"])
    test_set_human_predictions_transformed = affine_transform_channel_chart(human_test_data["cluster_positions"], test_set_human_predictions)
    metrics = CCEvaluation.compute_all_performance_metrics(test_set_human_predictions_transformed, human_test_data["cluster_positions"])
    CCEvaluation.plot_colorized(test_set_human_predictions_transformed, human_test_data["cluster_positions"], suptitle="PCC on Test Set (Human)", title=f"Unsupervised: MAE = {metrics['mae']:.3f}m", show=False, outfile=os.path.join(plots_output_dir, "pcc_unaug_human_scatter.jpg"))
    CCEvaluation.plot_error_ecdf(test_set_human_predictions_transformed, human_test_data["cluster_positions"], outfile=os.path.join(plots_output_dir, "pcc_unaug_human_ecdf.jpg"))
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.upper().rjust(6, ' ')}: {metric_value:.3f}")

    print("\nAugmented model")
    augmented_fcf_model = train_model(robot_training_data, augmented=True)
    
    print("\nTraining set robot")
    training_set_robot_predictions = augmented_fcf_model.predict(robot_training_data["cluster_features"]) + espargos_0007.centroid[:2]
    metrics = CCEvaluation.compute_all_performance_metrics(training_set_robot_predictions, robot_training_data["cluster_positions"])
    CCEvaluation.plot_colorized(training_set_robot_predictions, robot_training_data["cluster_positions"], suptitle="Augmented PCC on Training Set (Robot)", title=f"Unsupervised: MAE = {metrics['mae']:.3f}m", show=False, outfile=os.path.join(plots_output_dir, "pcc_aug_train_robot_scatter.png"))
    
    print("\nTest set robot")
    test_set_robot_predictions = augmented_fcf_model.predict(robot_test_data["cluster_features"]) + espargos_0007.centroid[:2]
    metrics = CCEvaluation.compute_all_performance_metrics(test_set_robot_predictions, robot_test_data["cluster_positions"])
    CCEvaluation.plot_colorized(test_set_robot_predictions, robot_test_data["cluster_positions"], suptitle="Augmented PCC on Test Set (Robot)", title=f"Unsupervised: MAE = {metrics['mae']:.3f}m", show=False, outfile=os.path.join(plots_output_dir, "pcc_aug_robot_scatter.png"))
    CCEvaluation.plot_error_ecdf(test_set_robot_predictions, robot_test_data["cluster_positions"], outfile=os.path.join(plots_output_dir, "pcc_aug_robot_ecdf.jpg"))
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.upper().rjust(6, ' ')}: {metric_value:.3f}")

    print("\nTest set human")
    test_set_human_predictions = augmented_fcf_model.predict(human_test_data["cluster_features"]) + espargos_0007.centroid[:2]
    metrics = CCEvaluation.compute_all_performance_metrics(test_set_human_predictions, human_test_data["cluster_positions"])
    CCEvaluation.plot_colorized(test_set_human_predictions, human_test_data["cluster_positions"], suptitle="Augmented PCC on Test Set (Human)", title=f"Unsupervised: MAE = {metrics['mae']:.3f}m", show=False, outfile=os.path.join(plots_output_dir, "pcc_aug_human_scatter.png"))
    CCEvaluation.plot_error_ecdf(test_set_human_predictions, human_test_data["cluster_positions"], outfile=os.path.join(plots_output_dir, "pcc_aug_human_ecdf.jpg"))
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.upper().rjust(6, ' ')}: {metric_value:.3f}")