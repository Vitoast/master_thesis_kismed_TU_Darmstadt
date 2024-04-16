import numpy as np
import matplotlib.pyplot as plt
import os
import re
import seaborn as sns
import pandas as pd
from collections import Counter


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

                    if feature_count < 5:
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
                stats_file.write(f'{feature_name}: Mean: {mean}, Median: {median}, Variance: {var}, Data Points: {len(cleaned_data)}, Min Value: {minimum}, Max Value: {maximum}\n')

