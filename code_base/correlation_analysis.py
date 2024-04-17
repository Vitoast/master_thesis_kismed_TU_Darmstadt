import os

from scipy import stats
import explorational_data_analysis as exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# Compute point biserial correlation for each clinical marker
def compute_marker_correlation(data_dictionary, output_directory):
    outcome_names, outcome_data = [], []
    marker_names, marker_data = [], []
    correlation_coefficients, p_values = [], []
    feature_count = 0
    for feature_name, feature_data in data_dictionary.items():
        cleaned_feature_name = exp.clean_feature_name(feature_name)
        if cleaned_feature_name.endswith('PRE') or cleaned_feature_name.endswith('POST'):
            marker_names.append(feature_name)
            marker_data.append(feature_data)
        elif feature_count < 5:
            outcome_names.append(feature_name)
            outcome_data.append(feature_data)
        feature_count += 1

    for outcome in range(len(outcome_names)):
        correlation_coefficients.append([])
        p_values.append([])
        # Check if the feature data contains only numeric values
        if all(isinstance(x, (int, float)) for x in outcome_data[outcome]):
            for marker in range(len(marker_names)):
                # Clean data from missing values
                # Find indices where either array1 or array2 is NaN
                nan_indices = np.logical_or(np.isnan(outcome_data[outcome]), np.isnan(marker_data[marker]))
                # Remove entries at nan_indices from both arrays
                markers_cleaned, outcomes_cleaned = [], []
                for nan_index in range(len(nan_indices)):
                    if not nan_indices[nan_index]:
                        markers_cleaned.append(outcome_data[outcome][nan_index])
                        outcomes_cleaned.append(marker_data[marker][nan_index])
                point_biserial_corr, p_value = stats.pointbiserialr(markers_cleaned, outcomes_cleaned)
                correlation_coefficients[outcome].append(point_biserial_corr)
                p_values[outcome].append(p_value)

            # Plot correlation coefficients
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            x = np.linspace(0, 49, 50)
            for i, (x_val, y1_val, y2_val) in enumerate(zip(x, correlation_coefficients[outcome], p_values[outcome])):
                color = np.random.rand(3, )
                if x_val > 24:
                    axs[1].plot([x_val, x_val], [0, y2_val], 'grey', marker='o', markersize=8)
                    axs[1].plot([x_val, x_val], [0, y1_val], color=color, marker='o', markersize=8,
                                label=marker_names[int(x_val)])
                else:
                    axs[0].plot([x_val, x_val], [0, y2_val], 'grey', marker='o', markersize=8)
                    axs[0].plot([x_val, x_val], [0, y1_val], color=color, marker='o', markersize=8,
                                label=marker_names[int(x_val)])

            # Plot horizontal lines for markers on the x-axis
            axs[0].hlines(0, x[0], x[-1], colors='black', linestyles='dashed', alpha=0.5)
            axs[1].hlines(0, x[0], x[-1], colors='black', linestyles='dashed', alpha=0.5)

            axs[0].set_title(f'Correlation of PRE-markers with {outcome_names[outcome]}')
            axs[0].set_xlabel('Marker')
            axs[0].set_ylabel('Correlation Coefficient (color) and P-Value (grey)')
            axs[0].legend(loc='upper right', fontsize='small')
            axs[1].set_title(f'Correlation of POST-markers with {outcome_names[outcome]}')
            axs[1].set_xlabel('Marker')
            axs[0].set_ylabel('Correlation Coefficient (color) and P-Value (grey)')
            axs[1].legend(loc='upper left', fontsize='small')
            # Save plot as PNG file
            cleaned_feature_name = exp.clean_feature_name(outcome_names[outcome])
            output_file_path = os.path.join(output_directory, f'Correlation of markers with {cleaned_feature_name}.jpg')
            plt.savefig(output_file_path)
            plt.close()
