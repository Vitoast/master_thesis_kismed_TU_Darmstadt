import os
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import code_base.global_variables as gl
import explorational_data_analysis as exp
from code_base.prediction import preprocess_data as pre


# Compute point biserial correlation for each clinical marker
def compute_marker_to_outcome_correlation(data_dictionary, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    outcome_names, outcome_data = [], []
    marker_names, marker_data = [], []
    all_feature_names, all_feature_data = [], []
    correlation_coefficients, p_values, sorted_features = [], [], []

    data_dictionary = pre.filter_data_sub_sets(data_dictionary)

    # Filter markers based on their belonging to POST or PRE
    feature_count = 0
    for feature_name, feature_data in data_dictionary.items():
        cleaned_feature_name = exp.clean_feature_name(feature_name)
        all_feature_names.append(cleaned_feature_name)
        all_feature_data.append(feature_data)
        # Check if the feature data contains only numeric values
        if all(isinstance(x, (int, float)) for x in feature_data):
            if feature_count < gl.number_outcomes:
                outcome_names.append(feature_name)
                outcome_data.append(feature_data)
                feature_count += 1
            else:
                marker_names.append(feature_name)
                marker_data.append(feature_data)

    # Create plots with correlation values for time series markers
    for outcome in range(gl.number_outcomes):
        correlation_coefficients.append([])
        p_values.append([])
        for marker in range(len(marker_names)):
            # Clean data from missing values
            # Find indices where either array1 or array2 is NaN
            nan_indices = np.logical_or(np.isnan(outcome_data[outcome]), np.isnan(marker_data[marker]))
            # Remove entries at nan_indices from both arrays
            markers_cleaned, outcomes_cleaned = [], []
            for nan_index in range(len(nan_indices)):
                if not nan_indices[nan_index]:
                    markers_cleaned.append(marker_data[marker][nan_index])
                    outcomes_cleaned.append(outcome_data[outcome][nan_index])
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
        sorted_features.append(marker_names_sorted)

        # Remove unusable data
        sorted_features[outcome], correlation_coefficients[outcome], p_values[outcome] = (
            zip(*[(v1, v2, v3) for v1, v2, v3 in
                  zip(sorted_features[outcome], correlation_coefficients[outcome], p_values[outcome])
                  if not (np.isnan(v2) or np.isnan(v3))]))

        plot_correlation = len(correlation_coefficients) < 70
        if plot_correlation:
            # Generate unique colors
            colors = [plt.cm.get_cmap('viridis', len(correlation_coefficients[outcome]))(i)
                      for i in range(len(correlation_coefficients[outcome]))]

            # Create the subplots, size depending on the number of features/ length of the legend
            if len(marker_names_sorted) > 101:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))
            else:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            # for descriptor in ['Correlation Coefficient', 'P-Value']:
            for i, (coeff, color, name) in enumerate(
                    zip(correlation_coefficients[outcome], colors, sorted_features[outcome])):
                ax1.scatter(i, coeff, color=color, label=name)
            ax1.set_title('Sorted Absolute Correlation Coefficients of Markers with ' + gl.outcome_descriptors[outcome])
            ax1.set_xlabel('Marker')
            ax1.set_ylabel('Correlation Coefficient')
            ax1.grid(True)

            # Scatter plot for sorted p-values
            for i, (pval, color, name) in enumerate(
                    zip(p_values[outcome], colors, sorted_features[outcome])):
                ax2.scatter(i, pval, color=color, label=name)
            ax2.set_title('P-Values Corresponding to Sorted Correlation Coefficients')
            ax2.set_xlabel('Marker')
            ax2.set_ylabel('P-Value')
            ax2.axhline(y=0.05, color='grey', linestyle='--', linewidth=2,
                        label='Significance threshold of P-Values (0.05)')
            ax2.grid(True)

            # Add legend besides plots
            fig.tight_layout()
            handles, labels = ax2.get_legend_handles_labels()
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 1, box.height])
            # Differentiate how much space legend takes
            if len(handles) > 101:
                fig.subplots_adjust(right=0.5)
                n_col = 3
            elif len(handles) > 51:
                fig.subplots_adjust(right=0.55)
                n_col = 2
            else:
                fig.subplots_adjust(right=0.7)
                n_col = 1
            plt.legend(handles, labels, bbox_to_anchor=(1.1, 2.2), loc='upper left', fontsize='small', ncol=n_col)

            # Save plot as PNG file
            cleaned_feature_name = exp.clean_feature_name(outcome_names[outcome])
            output_file_path = os.path.join(output_directory, f'Correlation of markers with {cleaned_feature_name}.pdf')
            plt.savefig(output_file_path, format='pdf')
            plt.close()

    # Save correlation coefficients for all features
    save_feature_importance(sorted_features, correlation_coefficients, p_values, output_directory)


# Compute the correlation between all features and return significant results (corr > 0.7, p-value < 0.05)
def compute_feature_collinearity(data_dictionary, result_path):
    # Only take predictive values in data set and save former config for later restoration
    former_data_set = gl.feature_blocks_to_use
    gl.feature_blocks_to_use = 'PRE_POST_BEFORE_DURING'
    data_dictionary = pre.filter_data_sub_sets(data_dictionary)
    test_data_dictionary = data_dictionary.copy()

    marker_names, marker_data = [], []
    correlation_pairs, correlation_values = [], []

    # Filter markers based on their belonging to POST or PRE
    feature_count = 0
    for feature_name, feature_data in data_dictionary.items():
        # Check if the feature data contains only numeric values
        if all(isinstance(x, (int, float)) for x in feature_data):
            if feature_count < gl.number_outcomes:
                feature_count += 1
            else:
                marker_names.append(feature_name)
                marker_data.append(feature_data)

    # Iterate over each marker of the whole set
    for marker_a in list(data_dictionary.keys()):
        # Remove the current marker from the test set so no double computations occur
        test_data_dictionary.pop(marker_a)
        # Iterate over each remaining marker
        for marker_b in list(test_data_dictionary.keys()):
            # Find indices where either array1 or array2 is NaN
            nan_indices = np.logical_or(np.isnan(data_dictionary[marker_a]), np.isnan(data_dictionary[marker_b]))
            # Remove entries at nan_indices from both arrays
            markers_cleaned, outcomes_cleaned = [], []
            for nan_index in range(len(nan_indices)):
                if not nan_indices[nan_index]:
                    markers_cleaned.append(data_dictionary[marker_a][nan_index])
                    outcomes_cleaned.append(data_dictionary[marker_b][nan_index])
            # Compute point-biserial correlation for the marker pair
            point_biserial_corr, p_value = stats.pointbiserialr(markers_cleaned, outcomes_cleaned)
            # If correlation is high and significant then save result
            if point_biserial_corr > 0.7 and p_value < 0.05:
                correlation_pairs.append([marker_a, marker_b])
                correlation_values.append([point_biserial_corr, p_value])
                print(marker_a, marker_b, point_biserial_corr, p_value)

    # Restore data set and return results
    gl.feature_blocks_to_use = former_data_set
    return correlation_pairs, correlation_values


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
    output_file_path = os.path.join(output_directory, f'Correlation matrix of clinical markers.pdf')
    plt.savefig(output_file_path, format='pdf')
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
                                            f'Pairwise plotting of clinical {category} markers regarding{cleaned_feature_name}.pdf')
            plt.savefig(output_file_path, format='pdf')
            plt.close()


# Save computed correlation coefficients in a file
def save_feature_importance(marker_names, correlation_coefficients, p_values, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    #  Do correlation analysis for each outcome
    for outcome in range(gl.number_outcomes):

        stats_file_path = os.path.join(output_directory,
                                       gl.outcome_descriptors[outcome] + '_correlation_coefficients_sorted.txt')

        # Open file to save results
        with open(stats_file_path, 'w') as stats_file:
            for feature_name, correlation, p_value in zip(marker_names[outcome], correlation_coefficients[outcome],
                                                          p_values[outcome]):
                stats_file.write(f'{feature_name},{correlation},{p_value},\n')


# Read in a SVD file that is produced by save_feature_importance()
def read_correlation_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            entries = line.strip().split(',')
            descriptor, cor_coeff, p_value = entries[0], float(entries[1]), float(entries[2])
            data.append((descriptor, cor_coeff, p_value))
    return data


# This is used to create a file that visualizes correlations to the outcomes
#   of the PRE and POST of cardiovascular markers based on prior correlation analysis
# Output: An excel file that maps each marker to each outcome and states only significant
#   correlations with their P-value and whether they are higher in PRE or POST
def sort_marker_correlation_data(result_path):
    combined_data = {}

    # Prepare output content
    excel_data = {'Descriptors': []}

    # Repeat for each outcome
    for outcome in gl.outcome_descriptors:
        # Get old corr compuations from file
        current_file_path = os.path.join(result_path, outcome + '_correlation_coefficients_sorted.txt')
        data = read_correlation_file(current_file_path)
        # Split between PRE and POST markers and remove suffix
        for descriptor, coeff, p_value in data:
            if 'PRE' in descriptor:
                base_descriptor = descriptor[:-3]
                suffix = descriptor[-3:]
            elif 'POST' in descriptor:
                base_descriptor = descriptor[:-4]
                suffix = descriptor[-4:]
            # Add current marker correlation if from PRE or POST
            if 'PRE' in descriptor or 'POST' in descriptor:
                if base_descriptor not in combined_data:
                    combined_data[base_descriptor] = {}
                if suffix not in combined_data[base_descriptor]:
                    combined_data[base_descriptor][suffix] = {}
                combined_data[base_descriptor][suffix][outcome] = (coeff, p_value)

        excel_data[f'Type_' + outcome] = []
        excel_data[f'Correlation Coefficient_' + outcome] = []
        excel_data[f'P-Value_' + outcome] = []

    # Now filter the markers form PRE and POST by their P-values
    for base_descriptor, suffixes in combined_data.items():
        excel_data['Descriptors'].append(base_descriptor)
        # Iterate over outcomes
        for outcome in gl.outcome_descriptors:
            # Initiate content with '-', stays if insignificant
            selected_type, selected_coeff, selected_pval = '-', '-', '-'

            # Check if value of PRE marker is significant
            if 'PRE' in suffixes and suffixes['PRE'].get(outcome) and suffixes['PRE'][outcome][1] <= 0.05:
                pre_coeff = suffixes['PRE'][outcome][0]
            else:
                pre_coeff = None

            # Check if value of POST marker is significant
            if 'POST' in suffixes and suffixes['POST'].get(outcome) and suffixes['POST'][outcome][1] <= 0.05:
                post_coeff = suffixes['POST'][outcome][0]
            else:
                post_coeff = None

            # Save PRE marker version if higher correlation or POST doesnt exist
            if pre_coeff is not None and (post_coeff is None or pre_coeff > post_coeff):
                selected_type = 'PRE'
                selected_coeff = suffixes['PRE'][outcome][0]
                selected_pval = suffixes['PRE'][outcome][1]
            # Save POST marker version if higher correlation and PRE doesnt exist
            elif post_coeff is not None:
                selected_type = 'POST'
                selected_coeff = suffixes['POST'][outcome][0]
                selected_pval = suffixes['POST'][outcome][1]

            excel_data[f'Type_' + outcome].append(selected_type)
            excel_data[f'Correlation Coefficient_' + outcome].append(selected_coeff)
            excel_data[f'P-Value_' + outcome].append(selected_pval)

    # Write results to new Excel file
    output_file_path = os.path.join(result_path, 'combined_ordered_markers.xlsx')
    with pd.ExcelWriter(output_file_path) as writer:
        pd.DataFrame(excel_data).to_excel(writer, index=False, sheet_name='SVD Data')
