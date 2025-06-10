import matplotlib.pyplot as plt
import sklearn.manifold
import sklearn.metrics
import numpy as np

def plot_colorized(positions, groundtruth_positions, suptitle = None, title = None, show = True, alpha = 1.0, outfile = None):
    """
    Creates a 2D scatter plot where points are colored based on their groundtruth position.
    This is used to generate the plots in Figure 4 of the source paper.
    """

    # Generate RGB colors for each datapoint based on its groundtruth position
    # The X coordinate maps to Red, Y to Blue, and distance from center to Green
    center_point = np.zeros(2, dtype = np.float32)
    center_point[0] = 0.5 * (np.min(groundtruth_positions[:, 0], axis = 0) + np.max(groundtruth_positions[:, 0], axis = 0))
    center_point[1] = 0.5 * (np.min(groundtruth_positions[:, 1], axis = 0) + np.max(groundtruth_positions[:, 1], axis = 0))
    NormalizeData = lambda in_data : (in_data - np.min(in_data)) / (np.max(in_data) - np.min(in_data))
    rgb_values = np.zeros((groundtruth_positions.shape[0], 3))
    rgb_values[:, 0] = 1 - 0.9 * NormalizeData(groundtruth_positions[:, 0])
    rgb_values[:, 1] = 0.8 * NormalizeData(np.square(np.linalg.norm(groundtruth_positions - center_point, axis=1)))
    rgb_values[:, 2] = 0.9 * NormalizeData(groundtruth_positions[:, 1])

     # Create the scatter plot
    plt.figure(figsize=(6, 6))        
    plt.scatter(positions[:, 0], positions[:, 1], c = rgb_values, alpha = alpha, s = 10, linewidths = 0)
    plt.axis("equal")


    # Formatting and saving
    if outfile is None:
        if suptitle is not None:
            plt.suptitle(suptitle, fontsize=14)
        if title is not None:
            plt.title(title, fontsize=16)
            plt.xlabel("x coordinate")
            plt.ylabel("y coordinate")
        if show:
            plt.show()
    else:
        plt.axis("off")
        xlim = (np.min(groundtruth_positions[:,0]) - 0.5, np.max(groundtruth_positions[:,0]) + 0.5)
        ylim = (np.min(groundtruth_positions[:,1]) - 0.5, np.max(groundtruth_positions[:,1]) + 0.5)
        #xlim = (np.min(positions[:,0]) - 0.5, np.max(positions[:,0]) + 0.5)
        #ylim = (np.min(positions[:,1]) - 0.5, np.max(positions[:,1]) + 0.5)
        plt.xlim(xlim)
        plt.ylim(ylim)
        print("xlim", xlim, "ylim", ylim)
        plt.savefig(outfile, bbox_inches = "tight", pad_inches = 0, transparent = True)

def compute_localization_metrics(estimated_positions, groundtruth_positions):
    """
    Computes basic localization error metrics.
    """

    errorvectors = groundtruth_positions - estimated_positions                 # Calculate the euclidean distance error for each point
    errors = np.sqrt(errorvectors[:,0]**2 + errorvectors[:,1]**2)
    mae = np.mean(errors)                                                      # Mean Absolute Error
    cep = np.median(errors)                                                    # Circular Error Probable (Median Error)

    return errorvectors, errors, mae, cep

def continuity(*args, **kwargs):
    """
    A wrapper to compute Continuity using sklearn's trustworthiness function.
    """
    # Continuity is Trustworthiness with the input arguments swapped.
    args = list(args)
    args[0], args[1] = args[1], args[0]                                         
    return sklearn.manifold.trustworthiness(*args, **kwargs)

def kruskal_stress(X, X_embedded, *, metric="euclidean"):
    """
    Computes the Kruskal's Stress metric.
    """ 
    # Computes pairwise distances in the original and embedded spaces
    dist_X = sklearn.metrics.pairwise_distances(X, metric = metric)
    dist_X_embedded = sklearn.metrics.pairwise_distances(X_embedded, metric = metric)
    # Finds the optimal scaling factor
    beta = np.divide(np.sum(dist_X * dist_X_embedded), np.sum(dist_X_embedded * dist_X_embedded))
    # Calculates the stress based on the difference between scaled distances
    return np.sqrt(np.divide(np.sum(np.square((dist_X - beta * dist_X_embedded))), np.sum(dist_X * dist_X)))

def ct_tw_ks_on_subset(groundtruth_positions, channel_chart_positions, downsampling = 10):
    """
    Computes Continuity, Trustworthiness, and Kruskal's Stress on a random subset of the data for efficiency.
    """
    # Select a random subset of indices.
    subset_indices = np.random.choice(range(len(groundtruth_positions)), len(groundtruth_positions) // downsampling)

    groundtruth_positions_subset = groundtruth_positions[subset_indices]
    channel_chart_positions_subset = channel_chart_positions[subset_indices]

    # Calculate metrics on the subset
    ct = continuity(groundtruth_positions_subset, channel_chart_positions_subset, n_neighbors = int(0.05 * len(groundtruth_positions_subset)))
    tw = sklearn.manifold.trustworthiness(groundtruth_positions_subset, channel_chart_positions_subset, n_neighbors = int(0.05 * len(groundtruth_positions_subset)))
    ks = kruskal_stress(groundtruth_positions_subset, channel_chart_positions_subset)

    return ct, tw, ks

def compute_all_performance_metrics(estimated_positions, groundtruth_positions):
    """
    Computes and returns a dictionary of all relevant performance metrics.
    """
    # Calculate euclidean distance errors.
    errorvectors = groundtruth_positions - estimated_positions
    errors = np.sqrt(errorvectors[:,0]**2 + errorvectors[:,1]**2)
    # Calculate localization error metrics.
    mae = np.mean(errors)
    drms = np.sqrt(np.mean(np.square(errors)))
    cep = np.median(errors)
    r95 = np.percentile(errors, 95)
    # Calculate chart quality metrics
    ct, tw, ks = ct_tw_ks_on_subset(estimated_positions, groundtruth_positions, downsampling = 1)

    return { "mae" : mae, "drms" : drms, "cep" : cep, "r95" : r95, "ks" : ks, "ct" : ct, "tw" : tw, }

def plot_error_ecdf(estimated_positions, groundtruth_positions, maxerr = 1.2, outfile = None):
    """
    Computes and plots the Empirical Cumulative Distribution Function (ECDF) of the localization errors. This is used to generate Figure 5 of the source paper.
    """
    errorvectors = groundtruth_positions - estimated_positions
    errors = np.sqrt(errorvectors[:,0]**2 + errorvectors[:,1]**2)

     # Compute the histogram of errors to get the probability density function (PDF)
    count, bins_count = np.histogram(errors, bins=200, range = (0, maxerr))
    pdf = count / len(errors)
    # Compute the cumulative distribution function (CDF) from the PDF
    cdf = np.cumsum(pdf)

    # Prepare data for plotting
    bins_count[0] = 0
    cdf = np.append([0], cdf)

    # Create the plot
    plt.figure(figsize=(5, 4))
    plt.plot(bins_count, cdf)
    plt.xlim((0, maxerr))
    plt.ylim((0, 1))
    plt.xlabel("Absolute Localization Error in m")
    plt.ylabel("Empirical CDF")
    plt.grid()

    if outfile is not None:
        plt.savefig(outfile, bbox_inches="tight")
        plt.close()
    else:
        plt.show()