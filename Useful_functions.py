import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def print_cm(cm, labels, directory=None, title="Full confusion matrix", color='GnBu', threshold=0.00):
    """
    Plot a confusion matrix with true labels on the y-axis.

    Args:
        cm (river confusion matrix): A confusion matrix to be plot.
        labels (list): List of class labels.
        title (string): Title of the plot.
        color (string): Plot color from the cmap colors.
        threshold (int): Threshold to display the value in the cm with annot=True.
       
    Returns:

    """

    num_labels = len(labels)

    cm_matrix = [[int(cm[true_label][pred_label]) for true_label in range(num_labels)] for pred_label in range(num_labels)]
    cm_matrix = np.array(cm_matrix).T
    row_sums = cm_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_matrix_perc = np.round(cm_matrix / row_sums, 2)
    label_names = [labels[label_num] for label_num in range(num_labels)]


    annot_matrix = np.empty_like(cm_matrix_perc, dtype=object)
    for i in range(cm_matrix_perc.shape[0]):
        for j in range(cm_matrix_perc.shape[1]):
            if cm_matrix_perc[i, j] >= threshold:
                annot_matrix[i, j] = f"{cm_matrix_perc[i, j]:.2f}"
            else:
                annot_matrix[i, j] = ""

    fig, ax = plt.subplots(figsize=(25, 25))

    sns.heatmap(cm_matrix_perc, annot=False, fmt='', xticklabels=label_names, yticklabels=label_names, cmap=color, 
                cbar=True, cbar_kws={'orientation': 'horizontal', 'pad': 0.2}, ax=ax, linewidths=.5)

    ax.set_xlabel('Predicted labels', fontsize=26)
    ax.set_ylabel('True labels', fontsize=26)
    ax.set_title('Confusion matrix', fontsize=28)
    ax.set_xticklabels(label_names, rotation=45, ha='right', fontsize=18)
    ax.set_yticklabels(label_names, rotation=0, fontsize=18)
    
    plt.subplots_adjust(bottom=0.2)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    plt.tight_layout()
    
    if directory:
        filename = f"{title}.eps"
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath, format='eps', dpi=600)

def print_means(sizes, accuracy, years, directory=None, title="of the chosen metric", idx=0):
    """
    Plot the mean accuracy over time.

    Args:
        sizes (list): List of cumulative observed samples across the years.
        accuracy (list): List of recorded metrics.
        years (list): List of observed years.
        title (string): Chosen metric to be included in the title.
        idx (int): 0 for accuracy, 1 for balanced accuracy, 3 for Cohen Kappa. 
       
    Returns:
        mean_acc (list): Mean chosen metric for each year of observation.
        full_mean (list): Mean chosen metric across the full set of samples.

    """
    min_length = min(len(sublist) for sublist in accuracy)
    truncated_list = [sublist[:min_length-1] for sublist in accuracy]
    res_array = np.array(truncated_list)
    cum_sizes_aux = [0]
    cum_sizes_aux.extend(sizes)
    mean_acc = []
    acc = res_array[:, idx]
    for i in range(len(sizes)):
        mean_acc.append(np.mean(acc[cum_sizes_aux[i]:cum_sizes_aux[i+1]]))
    mean_acc = np.array(mean_acc)
    full_mean = [mean_acc[i] * (cum_sizes_aux[i+1] - cum_sizes_aux[i]) for i in range(len(sizes))]
    full_mean = sum(full_mean) / sizes[-1]

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=years, y=mean_acc, marker='o')
    plt.xlabel('Year', fontsize=16)
    plt.ylabel(title.capitalize() if title != "of the chosen metric" else 'Metric', fontsize=16)
    plt.title('Mean ' + title + ' over years', fontsize=18)
    plt.axhline(full_mean, color='grey', linestyle='--', label=f'Overall Mean: {full_mean:.2f}')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower right', fontsize=13, framealpha=1)
    plt.grid(True)
    plt.tight_layout()

    if directory:
        filename = f"{title}.eps"
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath, format='eps', dpi=1200)

    return mean_acc, full_mean

def print_metrics(metrics, labels, sizes, years, directory=None):
    """
    Plot the given metrics over the increasing number of learned samples.

    Args:
        metrics (list of lists): A list of lists where each sublist contains metric values.
        labels (list): List of labels for each metric.
        sizes (list): List of cumulative observed samples across the years.

    Returns:

    """
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("muted", len(metrics))
    sns.lineplot(data=metrics)
    for x in sizes:
        plt.axvline(x=x, color='gray', linestyle='--')
    plt.xlabel('Observed samples', fontsize=16)
    plt.ylabel('Metric', fontsize=16)
    plt.ylim([0, 0.34])
    plt.title('Metrics over Observed Samples', fontsize=18)
    plt.axhline(metrics['Balanced accuracy'].mean(), color=palette[1], linestyle='dotted', 
                        label=f'Overall balanced accuracy for {labels} method: {metrics["Balanced accuracy"].mean():.2f}')
    plt.xticks(sizes, labels=years, fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower right', fontsize=13, framealpha=1)
    plt.grid(True)
    plt.tight_layout()

    if directory:
        filename = f"{labels}.eps"
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath, format='eps', dpi=1200)

def print_bal_acc_for_more_functions(metrics_list, labels, sizes, years, directory=None):
    """
    Plot the given metrics over the increasing number of learned samples.

    Args:
        metrics_list (list of lists of lists): A list of lists where each sublist contains metric values.
        labels (list of str): List of labels for each metric.
        sizes (list of int): List of cumulative observed samples across the years.
        years (list of int): List of years observed in the stream.
        directory (optional, str): String with the directory where the images should be saved.
    Returns:

    """
    plt.figure(figsize=(10, 6))
    for i, metrics in enumerate(metrics_list):
        min_length = min(len(sublist) for sublist in metrics)
        truncated_list = [sublist[:min_length-1] for sublist in metrics]
        res_array = np.array(truncated_list)
        plot_label=['Accuracy','Balanced accuracy']
        metrics = pd.DataFrame(res_array, columns=plot_label)

        palette = sns.color_palette("muted", len(metrics))
        sns.lineplot(data=metrics['Balanced accuracy'], 
                 label=f'{labels[i]} method with average balanced accuracy of {metrics["Balanced accuracy"].mean()*100:.0f}%')
        for x in sizes:
            plt.axvline(x=x, color='gray', linestyle='--')
    plt.xticks(sizes, labels=years, fontsize=14, rotation=45)
    plt.xlabel('Observed samples', fontsize=16)
    plt.ylabel('Balanced accuracy', fontsize=16)
    plt.ylim([0, 0.34])
    plt.title('Balanced accuracy per method', fontsize=18)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower right', fontsize=13, framealpha=1)
    plt.grid(True)
    plt.tight_layout()

    if directory:
        filename = f"{labels}.eps"
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath, format='eps', dpi=1200)

def compute_areas_under_curves(metrics_list, labels):
    """
    Compute the area under the Balanced Accuracy curve for each method.

    Args:
        metrics_list (list of lists of lists): A list where each item corresponds to a method,
                                               and contains multiple runs of metric values.
        labels (list of str): List of labels for each method.

    Returns:
        dict: A dictionary mapping each method label to its computed area under the curve.
    """
    area_dict = {}

    for i, method_metrics in enumerate(metrics_list):
        min_length = min(len(run) for run in method_metrics)
        truncated = [run[:min_length-1] for run in method_metrics]

        res_array = np.array(truncated)
        plot_label = ['Accuracy', 'Balanced accuracy']
        metrics = pd.DataFrame(res_array, columns=plot_label)

        balanced_acc = metrics['Balanced accuracy']
        x = np.arange(len(balanced_acc))
        area = np.trapz(balanced_acc, x)

        area_dict[labels[i]] = area

    return area_dict