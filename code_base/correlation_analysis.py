import os
import pandas as pd
from scipy import stats
import explorational_data_analysis as exp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import global_variables as gl


# Compute point biserial correlation for each clinical marker
def compute_marker_to_outcome_correlation(data_dictionary, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    outcome_names, outcome_data = [], []
    marker_names, marker_data = [], []
    all_feature_names, all_feature_data = [], []
    correlation_coefficients, p_values = [], []

    # Filter markers based on their belonging to POST or PRE
    feature_count = 0
    for feature_name, feature_data in data_dictionary.items():
        cleaned_feature_name = exp.clean_feature_name(feature_name)
        all_feature_names.append(cleaned_feature_name)
        all_feature_data.append(feature_data)
        if cleaned_feature_name.endswith('PRE') or cleaned_feature_name.endswith('POST'):
            marker_names.append(feature_name)
            marker_data.append(feature_data)
        elif feature_count < gl.number_outcomes and all(isinstance(x, (int, float)) for x in feature_data):
            outcome_names.append(feature_name)
            outcome_data.append(feature_data)
            feature_count += 1

    # Create plots with correlation values for time series markers
    for outcome in range(gl.number_outcomes):
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

            # Get absolute values, direction of correlation is not important
            correlation_coefficients[outcome] = [abs(c) for c in correlation_coefficients[outcome]]

            # Sort the correlation coefficients descending and reorder p-values accordingly
            sorted_indices = np.argsort(correlation_coefficients[outcome])[::-1]
            correlation_coefficients[outcome] = [correlation_coefficients[outcome][i] for i in sorted_indices]
            p_values[outcome] = [p_values[outcome][i] for i in sorted_indices]
            marker_names_sorted = [marker_names[i] for i in sorted_indices]

            # Generate unique colors
            colors = [plt.cm.get_cmap('viridis', len(correlation_coefficients[outcome]))(i) for i in range(len(correlation_coefficients[outcome]))]

            # Create the subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            # for descriptor in ['Correlation Coefficient', 'P-Value']:
            for i, (coeff, color, name) in enumerate(zip(correlation_coefficients[outcome], colors, marker_names_sorted)):
                ax1.scatter(i, coeff, color=color, label=name)
            ax1.set_title('Sorted Absolute Correlation Coefficients of Markers with ' + gl.outcome_descriptors[outcome])
            ax1.set_xlabel('Marker')
            ax1.set_ylabel('Correlation Coefficient')

            # Scatter plot for sorted p-values
            for i, (pval, color) in enumerate(zip(p_values[outcome], colors)):
                ax2.scatter(i, pval, color=color)
            ax2.set_title('P-Values Corresponding to Sorted Correlation Coefficients')
            ax2.set_xlabel('Marker')
            ax2.set_ylabel('P-Value')

            # Add legend besides plots
            fig.tight_layout()
            handles, labels = ax1.get_legend_handles_labels()
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 1, box.height])
            fig.subplots_adjust(right=0.7)
            plt.legend(handles, labels, bbox_to_anchor=(1.1, 2.2), loc='upper left', fontsize='small')

            # Save plot as PNG file
            cleaned_feature_name = exp.clean_feature_name(outcome_names[outcome])
            output_file_path = os.path.join(output_directory, f'Correlation of markers with {cleaned_feature_name}.jpg')
            plt.savefig(output_file_path)
            plt.close()

    # Save correlation coefficients for all features
    # save_feature_importance(outcome_names, outcome_data, all_feature_names, all_feature_data, output_directory)


# Compute and plot the correlation matrix of all PRE- and POST-surgery markers
def compute_marker_correlation_matrix(data_dictionary, output_directory):
    marker_names, marker_data = [], []
    for feature_name, feature_data in data_dictionary.items():
        cleaned_feature_name = exp.clean_feature_name(feature_name)
        if cleaned_feature_name.endswith('PRE') or cleaned_feature_name.endswith('POST'):
            marker_names.append(feature_name)
            marker_data.append(feature_data)

    # Create a pandas DataFrame
    df = pd.DataFrame(np.transpose(marker_data), columns=marker_names)

    # Compute the correlation matrix
    corr_matrix = df.corr(method='spearman')

    # Plot the correlation matrix using a heatmap
    plt.figure(figsize=(30, 25))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f")
    plt.title('Correlation Matrix of Clinical Markers')
    # Save plot as PNG file
    output_file_path = os.path.join(output_directory, f'Correlation matrix of clinical markers.jpg')
    plt.savefig(output_file_path)
    plt.close()


def show_pairwise_marker_correlation(data_dictionary, output_directory):
    # Do pairwise correlation only for members of PRE and POST
    for category in ['PRE', 'POST']:
        for outcome in range(1, gl.number_outcomes):
            marker_names, marker_data = [], []
            counter = 0
            for feature_name, feature_data in data_dictionary.items():
                if counter == outcome:
                    marker_data.append(feature_data)
                    marker_names.append(feature_name)
                cleaned_feature_name = exp.clean_feature_name(feature_name)
                if cleaned_feature_name.endswith(category):
                    marker_names.append(feature_name)
                    marker_data.append(feature_data)
                counter += 1
            # Create a pandas DataFrame
            df = pd.DataFrame(np.transpose(marker_data), columns=marker_names)

            sns.pairplot(df, vars=df.columns[1:], kind='kde', plot_kws=dict(levels=2), hue=marker_names[0])
            # Save plot as PNG file
            cleaned_feature_name = exp.clean_feature_name(marker_names[0])
            output_file_path = os.path.join(output_directory,
                                            f'Pairwise plotting of clinical {category} markers regarding{cleaned_feature_name}.jpg')
            plt.savefig(output_file_path)
            plt.close()


def save_feature_importance(outcome_names, outcome_data, all_feature_names, all_feature_data, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    stats_file_path = os.path.join(output_directory, 'correlation_coefficients.txt')
    correlation_coefficients, p_values = [], []
    sorted_features_result = []

    # Open file to save results
    with open(stats_file_path, 'w') as stats_file:

        #  Do correlation analysis for each outcome
        for outcome in range(len(outcome_names)):
            correlation_coefficients.append([])
            p_values.append([])
            # Compute correlation between outcome and feature
            for feature in all_feature_names:
                # Check if the feature data contains only numeric values
                if all(isinstance(x, (int, float)) for x in all_feature_data[outcome]):
                    point_biserial_corr, p_value = stats.pointbiserialr(feature, outcome_data[outcome])
                    correlation_coefficients[outcome].append(point_biserial_corr)
                    p_values[outcome].append(p_value)
                else:
                    all_feature_names.pop(all_feature_names.index(feature))

            # Tie features together with their importance and print the sorted list, leaving out neglected features
            features_with_correlation = list(zip(all_feature_names, correlation_coefficients[outcome], p_values[outcome]))
            features_with_correlation_sorted = sorted(features_with_correlation, key=lambda x: x[1], reverse=True)
            sorted_features_result.append(features_with_correlation_sorted)
            stats_file.write(f'Sorted correlation coefficients for {outcome_names[outcome]},\n')
            for feature_name, correlation, p_value in features_with_correlation_sorted:
                stats_file.write(f'{feature_name},{correlation},{p_value},\n')

    # return computed results
    return sorted_features_result
