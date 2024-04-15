import numpy as np
import matplotlib.pyplot as plt
import os
import re


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
    outcome_counter = 0
    with open(stats_file_path, 'w') as stats_file:
        # Iterate over each feature
        for feature_name, feature_data in data_dictionary.items():
            # Check if the feature data contains only numeric values
            if all(isinstance(x, (int, float)) for x in feature_data):
                # Clean data from nan
                cleaned_data = [x for x in feature_data if not np.isnan(x)]

                # Plot a histogram of the data
                plt.hist(cleaned_data, bins=100, color='green', edgecolor='black')
                plt.title(f'Histogram of feature {feature_name}')
                plt.xlabel('Value')
                plt.ylabel('Occurrences')
                plt.grid(True)
                plt.tight_layout()

                cleaned_feature_name = clean_feature_name(feature_name)

                # Determine subfolder based on feature name
                if outcome_counter < 5:
                    subfolder = outcome_directory
                else:
                    if cleaned_feature_name.endswith('PRE') or cleaned_feature_name.endswith('POST'):
                        subfolder = clinical_marker_directory
                    else:
                        subfolder = clinical_data_directory

                # Save histogram as PNG file
                output_file_path = os.path.join(subfolder, f'hist_{cleaned_feature_name}.png')
                plt.savefig(output_file_path)
                plt.close()

                # Calculate mean and variance
                mean = np.mean(cleaned_data)
                var = np.var(cleaned_data)

                # Save mean and variance to text file
                stats_file.write(f'{feature_name}: Mean = {mean}, Variance = {var}, Data Points: {len(cleaned_data)}\n')

            outcome_counter += 1
