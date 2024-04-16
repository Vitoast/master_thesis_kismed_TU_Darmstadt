from scipy import stats
import explorational_data_analysis as exp
import numpy as np


# Compute point biserial correlation for each clinical marker
def compute_marker_correlation(data_dictionary, output_directory):

    outcome_names, outcome_data = [], []
    marker_names, marker_data = [], []
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

    for marker in range(len(marker_names)):
        print(f'Correlation with {marker_names[marker]}\n')
        for outcome in range(len(outcome_names)):
            # Check if the feature data contains only numeric values
            if all(isinstance(x, (int, float)) for x in outcome_data[outcome]):
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
                print(f"   Point biserial correlation coefficient for {marker_names[marker]}:", point_biserial_corr)
                print("   p-value:", p_value)
