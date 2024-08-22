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
                # Histograms are outcome dependent
                for outcome in range(gl.number_outcomes):
                    # Clean data from nan
                    # Create the cleaned data by filtering out NaN values
                    cleaned_data = [(x, label) for x, label in
                                    zip(feature_data, data_dictionary[gl.original_outcome_strings[outcome]]) if
                                    not np.isnan(x)]

                    # Unpack the cleaned data into separate arrays
                    cleaned_x_values, cleaned_labels = zip(*cleaned_data)

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
                        plt.xlabel('Outcome')
                        plt.ylabel('Occurences')

                        if feature_count < gl.number_outcomes:
                            plt.title(f'Ratio of {gl.outcome_descriptors[feature_count]}')

                            output_file_path = os.path.join(outcome_directory, f'bar_{cleaned_feature_name}.png')
                        else:
                            plt.title(f'Ratio of {feature_name}')
                            output_file_path = os.path.join(clinical_data_directory, f'bar_{cleaned_feature_name}.png')

                    else:
                        plt.figure(figsize=(8, 4))

                        data_frame = pd.DataFrame({
                            'Value': np.array(cleaned_x_values),
                            'Outcome (1=yes)': np.array(cleaned_labels)
                        })

                        # Create histograms using seaborn
                        sns.histplot(data=data_frame, x='Value', hue='Outcome (1=yes)', multiple='dodge', bins=40,
                                     kde=True, edgecolor='black', alpha=0.5)

                        # sns.histplot(cleaned_data, bins=20, kde=True, color='green', edgecolor='black')
                        plt.title(f'Histogram of feature {feature_name} for outcome {gl.outcome_descriptors[outcome]}')
                        plt.xlabel('Value')
                        plt.ylabel('Occurrences')
                        plt.grid(True)
                        plt.tight_layout()

                        if cleaned_feature_name.endswith('PRE') or cleaned_feature_name.endswith('POST'):
                            output_file_path = os.path.join(clinical_marker_directory,
                                                            f'hist_{cleaned_feature_name}_{gl.outcome_descriptors[outcome]}.svg')
                        else:
                            output_file_path = os.path.join(clinical_data_directory,
                                                            f'hist_{cleaned_feature_name}_{gl.outcome_descriptors[outcome]}.svg')

                    # Save plot as svg file
                    plt.savefig(output_file_path, format='svg')
                    plt.close()
                    feature_count += 1

                    # Calculate mean and variance
                    mean = np.mean(cleaned_x_values)
                    var = np.var(cleaned_x_values)
                    median = np.median(cleaned_x_values)
                    minimum = min(cleaned_data)
                    maximum = max(cleaned_data)

                    # Save mean and variance to text file
                    stats_file.write(
                        f'{feature_name}: Mean: {mean}, Median: {median}, Variance: {var}, Data Points: {len(cleaned_x_values)}, Min Value: {minimum}, Max Value: {maximum}\n')


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

    # Set of reasonable UMAP hyperparameters
    n_neighbors_range = [5, 10, 20, 50]
    min_dist_range = [0.001, 0.01, 0.1, 0.5]

    # Preprocess data per outcome separately
    for outcome in range(gl.number_outcomes):
        preprocessed_train_data, preprocessed_test_data = pre.preprocess_data(data_dictionary.copy(),
                                                                              data_dictionary.copy(),
                                                                              outcome,
                                                                              standardize=gl.standardize,
                                                                              impute='median_group',
                                                                              z_score_threshold=0,
                                                                              oversample_rate=0)
        x_train, x_test, y_train, y_test = cl.split_maps(preprocessed_train_data, [])
        x_data_per_outcome.append(x_train)
        y_data_per_outcome.append(y_train)

    # Try out every combination of the hyperparameters
    for n_neighbors in n_neighbors_range:
        for min_dist in min_dist_range:

            current_output_directory = os.path.join(output_directory,
                                                    'Test_' + str(n_neighbors_range.index(n_neighbors))
                                                    + '_' + str(min_dist_range.index(min_dist)))
            os.makedirs(current_output_directory, exist_ok=True)

            # Plot umap for each outcome separately
            for outcome in range(gl.number_outcomes):
                result_path = os.path.join(current_output_directory, f'umap_of_only_{gl.outcome_descriptors[outcome]}')

                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
                embedding = reducer.fit_transform(np.reshape(x_data_per_outcome[outcome],
                                                             (len(x_data_per_outcome[outcome][0]),
                                                              len(x_data_per_outcome[outcome]))))
                fig, ax = plt.subplots(1, figsize=(8, 6))
                plt.scatter(*embedding.T, s=20, c=y_data_per_outcome[outcome], cmap='viridis', alpha=1.0)
                plt.setp(ax, xticks=[], yticks=[])
                cbar = plt.colorbar(boundaries=np.arange(3) - 0.5)
                cbar.set_ticks(np.arange(2))
                classes = ['Not ' + gl.outcome_descriptors[outcome], gl.outcome_descriptors[outcome]]
                cbar.set_ticklabels(classes)
                plt.title(gl.outcome_descriptors[outcome] + ' UMAP visualization')
                plt.tight_layout()
                plt.savefig(result_path, format='svg')
                plt.close()

            # Plot umap of each combination of outcomes
            for outcome_a in range(gl.number_outcomes - 1):
                for outcome_b in range(gl.number_outcomes - 1):
                    # Skip doubled calculations
                    if outcome_b <= outcome_a:
                        continue

                    result_path = os.path.join(current_output_directory,
                                               f'umap_of_{gl.outcome_descriptors[outcome_a]}_with_{gl.outcome_descriptors[outcome_b]}')

                    # Calculate independent and overlapping classes
                    y_tmp = np.sum(y_data_per_outcome[outcome_a] + 2 * y_data_per_outcome[outcome_b], axis=0)
                    # Calculate and plot umap
                    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
                    embedding = reducer.fit_transform(np.reshape(x_data_per_outcome[outcome_a],
                                                                 (len(x_data_per_outcome[outcome_a][0]),
                                                                  len(x_data_per_outcome[outcome_a]))))
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
                    plt.tight_layout()
                    plt.savefig(result_path, format='svg')
                    plt.close()

            # Plot UMAP for all outcomes
            result_path = os.path.join(current_output_directory, f'umap_of_all_outcomes')

            # Define which class combinations are occurring in the data set
            classes = ['No adverse event',
                       'Only AKD',
                       'Only AF',
                       'AKD and LCOS',
                       'AKD and AF',
                       'AKI1 and AF',
                       'AKD, LCOS and AF',
                       'AKI1, LCOS and AF']
            # Map the computed scalars to the class combinations
            # (This is done to not plot not existing combinations in the legend)
            class_mapping = {0: 0, 1: 1, 4: 2, 5: 3, 9: 4, 11: 5, 13: 6, 15: 7}
            # Scalar values for each outcome to distinguish overlapping classes
            scalars = np.array([1, 2, 4, 8])
            # Sum the scaled outcomes to distinguish between combinations of outcomes
            y_tmp = np.sum(scalar * array for scalar, array in zip(scalars, np.array(y_data_per_outcome)))
            y_scaled = []
            # Map values to indices of class descriptors
            for y_val in y_tmp[0]:
                y_scaled.append(class_mapping[y_val])

            # Calculate and plot umap
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
            embedding = reducer.fit_transform(np.reshape(x_data_per_outcome[0],
                                                         (len(x_data_per_outcome[0][0]),
                                                          len(x_data_per_outcome[0]))))
            # Plot
            fig, ax = plt.subplots(1, figsize=(8, 6))
            plt.scatter(*embedding.T, s=40, c=y_scaled, cmap='nipy_spectral', alpha=1.0)
            plt.setp(ax, xticks=[], yticks=[])
            cbar = plt.colorbar(boundaries=np.arange(9) - 0.5)
            cbar.set_ticks(np.arange(9))
            cbar.set_ticklabels(classes)
            plt.title('UMAP visualization of all outcomes and their combinations')
            plt.tight_layout()
            plt.savefig(result_path, format='svg')
            plt.close()
