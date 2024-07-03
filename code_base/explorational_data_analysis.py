import numpy as np
import matplotlib.pyplot as plt
import os
import re
import seaborn as sns
import pandas as pd
from collections import Counter
import umap

import global_variables as gl
import preprocess_data as pre
import classification as cl


def clean_feature_name(feature_name):
    # Remove invalid characters from feature name for Windows file system
    cleaned_name = re.sub(r'[<>:"/\\|?*]', '_', feature_name)
    return cleaned_name


def explore_data(data_dictionary, output_directory):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    clinical_marker_directory = os.path.join(output_directory, 'clinical_marker')
    clinical_data_directory = os.path.join(output_directory, 'clinical_data')
    outcome_directory = os.path.join(output_directory, 'outcomes')
    os.makedirs(clinical_marker_directory, exist_ok=True)
    os.makedirs(clinical_data_directory, exist_ok=True)
    os.makedirs(outcome_directory, exist_ok=True)

    # Create file to store results
    stats_file_path = os.path.join(output_directory, 'exploration_results.txt')
    feature_count = 0
    with open(stats_file_path, 'w') as stats_file:
        # Iterate over each feature
        for feature_name, feature_data in data_dictionary.items():
            # Check if the feature data contains only numeric values
            if all(isinstance(x, (int, float)) for x in feature_data):
                # Clean data from nan
                cleaned_data = [x for x in feature_data if not np.isnan(x)]

                cleaned_feature_name = clean_feature_name(feature_name)

                classification = all(x in [0, 1] for x in cleaned_data)

                # Determine subfolder based on feature name
                if classification:
                    counts = Counter(cleaned_data)
                    labels = ['No', 'Yes']
                    sizes = [counts[0], counts[1]]

                    # Creating the pie chart
                    plt.figure(figsize=(8, 5))
                    plt.bar(labels, sizes, color=['green', 'red'])
                    plt.title(f'Ratio of {feature_name}')
                    plt.xlabel('Outcome')
                    plt.ylabel('Occurences')

                    if feature_count < gl.number_outcomes:
                        output_file_path = os.path.join(outcome_directory, f'bar_{cleaned_feature_name}.png')
                    else:
                        output_file_path = os.path.join(clinical_data_directory, f'bar_{cleaned_feature_name}.png')

                else:
                    # Plot a histogram of the data
                    plt.figure(figsize=(8, 4))
                    sns.histplot(cleaned_data, bins=20, kde=True, color='green', edgecolor='black')
                    plt.title(f'Histogram of feature {feature_name}')
                    plt.xlabel('Value')
                    plt.ylabel('Occurrences')
                    plt.grid(True)
                    plt.tight_layout()

                    if cleaned_feature_name.endswith('PRE') or cleaned_feature_name.endswith('POST'):
                        output_file_path = os.path.join(clinical_marker_directory, f'hist_{cleaned_feature_name}.png')
                    else:
                        output_file_path = os.path.join(clinical_data_directory, f'hist_{cleaned_feature_name}.png')

                        # Save plot as PNG file
                plt.savefig(output_file_path)
                plt.close()
                feature_count += 1

                # Calculate mean and variance
                mean = np.mean(cleaned_data)
                var = np.var(cleaned_data)
                median = np.median(cleaned_data)
                minimum = min(cleaned_data)
                maximum = max(cleaned_data)

                # Save mean and variance to text file
                stats_file.write(
                    f'{feature_name}: Mean: {mean}, Median: {median}, Variance: {var}, Data Points: {len(cleaned_data)}, Min Value: {minimum}, Max Value: {maximum}\n')


# Check that the data sets are suitable for classification
def check_data_sets(train_map, test_map):
    for (feature_name_train, feature_data_train), (feature_name_test, feature_data_test) in zip(train_map.items(),
                                                                                                test_map.items()):
        # Check if data is usable for classification
        if not np.isfinite(feature_data_train).all():
            print(feature_name_train, " in train set is not formatted for classification.")
        if not np.isfinite(feature_data_test).all():
            print(feature_name_test, " in test set is not formatted for classification.")
        # Check if features are at the same place
        if feature_name_train != feature_name_test:
            print("The features in the sets do not match: ", feature_name_train, " ", feature_name_test)
            break


# Visualize clustering of the data with umap
def plot_umap(data_dictionary, output_directory):
    x_data_per_outcome, y_data_per_outcome = [], []
    os.makedirs(output_directory, exist_ok=True)

    # Preprocess data per outcome separately
    for outcome in range(gl.number_outcomes):
        preprocessed_train_data, preprocessed_test_data = pre.preprocess_data(data_dictionary.copy(),
                                                                              data_dictionary.copy(),
                                                                              outcome,
                                                                              standardize=gl.standardize,
                                                                              impute='median_std',
                                                                              z_score_threshold=0,
                                                                              oversample_rate=0)
        x_train, x_test, y_train, y_test = cl.split_maps(preprocessed_train_data, [])
        x_data_per_outcome.append(x_train)
        y_data_per_outcome.append(y_train)

    # Plot umap for each outcome separately
    for outcome in range(gl.number_outcomes):
        result_path = os.path.join(output_directory, f'umap_of_only_{gl.outcome_descriptors[outcome]}')

        embedding = umap.UMAP().fit_transform(np.reshape(x_data_per_outcome[outcome],
                                                         (len(x_data_per_outcome[outcome][0]),
                                                          len(x_data_per_outcome[outcome]))),
                                              y_data_per_outcome[outcome][0])
        fig, ax = plt.subplots(1, figsize=(8, 6))
        plt.scatter(*embedding.T, s=20, c=y_data_per_outcome[outcome], cmap='viridis', alpha=1.0)
        plt.setp(ax, xticks=[], yticks=[])
        cbar = plt.colorbar(boundaries=np.arange(3) - 0.5)
        cbar.set_ticks(np.arange(2))
        classes = ['Not ' + gl.outcome_descriptors[outcome], gl.outcome_descriptors[outcome]]
        cbar.set_ticklabels(classes)
        plt.title(gl.outcome_descriptors[outcome] + ' UMAP visualization')
        plt.tight_layout()
        plt.savefig(result_path)
        plt.close()

    # Plot umap of each combination of outcomes
    for outcome_a in range(gl.number_outcomes - 1):
        for outcome_b in range(gl.number_outcomes - 1):
            # Skip doubled calculations
            if outcome_b <= outcome_a:
                continue

            result_path = os.path.join(output_directory,
                                       f'umap_of_{gl.outcome_descriptors[outcome_a]}_with_{gl.outcome_descriptors[outcome_b]}')

            # Calculate independent and overlapping classes
            y_tmp = np.sum(y_data_per_outcome[outcome_a] + 2 * y_data_per_outcome[outcome_b], axis=0)
            # Calculate and plot umap
            embedding = umap.UMAP().fit_transform(np.reshape(x_data_per_outcome[outcome_a],
                                                             (len(x_data_per_outcome[outcome_a][0]),
                                                              len(x_data_per_outcome[outcome_a]))),
                                                  y_tmp)
            # Plot
            fig, ax = plt.subplots(1, figsize=(8, 6))
            plt.scatter(*embedding.T, s=40, c=y_tmp, cmap='viridis', alpha=1.0)
            plt.setp(ax, xticks=[], yticks=[])
            cbar = plt.colorbar(boundaries=np.arange(5) - 0.5)
            cbar.set_ticks(np.arange(4))
            classes = ['No adverse event', 'Only ' + gl.outcome_descriptors[outcome_a],
                       'Only ' + gl.outcome_descriptors[outcome_b],
                       gl.outcome_descriptors[outcome_a] + ' and ' + gl.outcome_descriptors[outcome_b]]
            cbar.set_ticklabels(classes)
            plt.title(gl.outcome_descriptors[outcome_a] + ' and ' + gl.outcome_descriptors[outcome_b]
                      + ' UMAP visualization')
            plt.legend()
            plt.tight_layout()
            plt.savefig(result_path)
            plt.close()

    # Plot UMAP for all outcomes
    result_path = os.path.join(output_directory, f'umap_of_all_outcomes')

    # Scalar values for each outcome to distinguish overlapping classes
    scalars = np.array([0, 1, 4, 8])
    # Sum the scaled outcomes
    y_tmp = np.sum(scalar * array for scalar, array in zip(scalars, np.array(y_data_per_outcome)))
    # y_tmp = y_data_per_outcome[0] + (y_data_per_outcome[1] * 2) + (y_data_per_outcome[2] * 4) + (y_data_per_outcome[3] * 8)
    # Calculate and plot umap
    embedding = umap.UMAP().fit_transform(np.reshape(x_data_per_outcome[0],
                                                     (len(x_data_per_outcome[0][0]),
                                                      len(x_data_per_outcome[0]))),
                                          y_tmp[0])
    # Plot
    fig, ax = plt.subplots(1, figsize=(8, 6))
    plt.scatter(*embedding.T, s=40, c=y_tmp, cmap='nipy_spectral', alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(14) - 0.5)
    cbar.set_ticks(np.arange(13))
    classes = ['No adverse event',
               'Only ' + gl.outcome_descriptors[0],
               'Only ' + gl.outcome_descriptors[1],
               gl.outcome_descriptors[0] + ' and ' + gl.outcome_descriptors[1],
               'Only ' + gl.outcome_descriptors[2],
               gl.outcome_descriptors[0] + ' and ' + gl.outcome_descriptors[2],
               gl.outcome_descriptors[1] + ' and ' + gl.outcome_descriptors[2],
               gl.outcome_descriptors[0] + ', ' + gl.outcome_descriptors[1] + ' and ' + gl.outcome_descriptors[2],
               'Only ' + gl.outcome_descriptors[3],
               gl.outcome_descriptors[0] + ' and ' + gl.outcome_descriptors[3],
               gl.outcome_descriptors[1] + ' and ' + gl.outcome_descriptors[3],
               gl.outcome_descriptors[0] + ', ' + gl.outcome_descriptors[1] + ' and ' + gl.outcome_descriptors[3],
               gl.outcome_descriptors[2] + ' and ' + gl.outcome_descriptors[3],
               gl.outcome_descriptors[0] + ', ' + gl.outcome_descriptors[1] + ' and ' + gl.outcome_descriptors[3]]
               # gl.outcome_descriptors[0] + ', ' + gl.outcome_descriptors[2] + ' and ' + gl.outcome_descriptors[3],
               # gl.outcome_descriptors[1] + ', ' + gl.outcome_descriptors[2] + ' and ' + gl.outcome_descriptors[3],
               # gl.outcome_descriptors[0] + ', ' + gl.outcome_descriptors[1] + ', '
               # + gl.outcome_descriptors[2] + ' and ' + gl.outcome_descriptors[3]]
    cbar.set_ticklabels(classes)
    plt.title('UMAP visualization of all outcomes and their combinations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_path)
    plt.close()
