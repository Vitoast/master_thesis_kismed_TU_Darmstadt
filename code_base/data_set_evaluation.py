import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_classif
from scipy import stats

import global_variables as gl
import preprocess_data as pre
import parameter_evaluation as pe
import classification as clf
import feature_evaluation as fe


# This computes the mutual information (MI) between subsets of the data set and the whole data set
def compare_subset_information_gain(complete_data_map, result_path):
    mutual_info_gains = []

    # Calculate MI for each outcome
    for outcome_target_index in range(gl.number_outcomes):

        # Preprocess combined data set to get reference
        gl.feature_blocks_to_use = 'PRE_POST_BEFORE_DURING'
        preprocessed_complete_set = pre.preprocess_data(complete_data_map, complete_data_map,
                                                        outcome_target_index,
                                                        standardize=True,
                                                        z_score_threshold=0,
                                                        oversample_rate=0)[0]
        # Delete target field of outcome, save it and convert remaining set to panda DF
        target_values = pd.DataFrame(preprocessed_complete_set.pop(gl.original_outcome_strings[outcome_target_index]))
        preprocessed_complete_set_df = pd.DataFrame(preprocessed_complete_set)

        # Consider PRE and POST markers against BEFORE and DURING features
        for current_subset in ['PRE_POST', 'BEFORE_DURING']:
            gl.feature_blocks_to_use = current_subset
            # Preprocess data with scaling and imputation
            preprocessed_subset = pre.preprocess_data(complete_data_map, complete_data_map,
                                                      outcome_target_index,
                                                      standardize=True,
                                                      impute='median_group',
                                                      z_score_threshold=0,
                                                      oversample_rate=0)[0]
            # Delete target field of outcome and convert to panda df
            preprocessed_subset.pop(gl.original_outcome_strings[outcome_target_index])
            preprocessed_subset_df = pd.DataFrame(preprocessed_subset)

            # Compute MI for each set and then the difference as gain
            mutual_info_complete_set = mutual_info_classif(preprocessed_complete_set_df, target_values)
            mutual_info_subset = mutual_info_classif(preprocessed_subset_df, target_values)
            mutual_info_gain = np.sum(mutual_info_complete_set) - np.sum(mutual_info_subset)
            mutual_info_gains.append(mutual_info_gain)

            print(f'{gl.outcome_descriptors[outcome_target_index]} conditional mutual information (CMI)\n'
                  f'MI: complete set to outcome: {np.sum(mutual_info_complete_set)}\n'
                  f'MI: contribution subset {current_subset} to outcome: {np.sum(mutual_info_subset)}\n'
                  f'CMI: contribution other subset to complete set: {mutual_info_gain}\n')

    return mutual_info_gains


# Analyse the difference between the PRE and POST markers
# Models are trained with each single feature and the performance for both sets is saved in an Excel file per outcome
def compare_pre_to_post_marker_performance(complete_data_map, result_path):
    os.makedirs(result_path, exist_ok=True)
    result_file_path = os.path.join(result_path, 'pre_post_single_performance.xlsx')

    # Prepare data structures that hold outcomes of study
    accuracies_per_outcome_pre, f1_scores_per_outcome_pre = pe.create_result_structure()
    accuracy_var_per_outcome_pre, f1_score_var_per_outcome_pre = pe.create_result_structure()
    # Prepare data structures to hold variance
    accuracies_per_outcome_post, f1_scores_per_outcome_post = pe.create_result_structure()
    accuracy_var_per_outcome_post, f1_score_var_per_outcome_post = pe.create_result_structure()

    # Get features for both sets
    gl.feature_blocks_to_use = 'PRE'
    pre_marker_map = pre.filter_data_sub_sets(complete_data_map)
    pre_marker_single_map = {}
    gl.feature_blocks_to_use = 'POST'
    post_marker_map = pre.filter_data_sub_sets(complete_data_map)
    post_marker_single_map = {}

    # Set global data set variable so PRE and POST does not get kicked out later
    gl.feature_blocks_to_use = 'PRE_POST'

    # Then perform classification with all given models and evaluate the performance with f1 score
    for outcome_value in range(gl.number_outcomes):
        result_file_path = os.path.join(result_path, gl.outcome_descriptors[outcome_value]
                                        + '_pre_post_single_performance.xlsx')

        for model in gl.classifiers:
            for (pre_key, pre_data), (post_key, post_data) in zip(pre_marker_map.items(), post_marker_map.items()):

                pre_marker_single_map[pre_key] = pre_data
                post_marker_single_map[post_key] = post_data

                # If current feature is an outcome, do not classify
                if pre_key in gl.original_outcome_strings or post_key in gl.original_outcome_strings:
                    continue

                # Train and predict with k-fold validation for PRE
                accuracy_results, accuracy_var, f1_scores, f1_var, tmp0, tmp1 = clf.classify_k_fold(
                    pre_marker_single_map,
                    outcome_value,
                    result_path, [],
                    model, False, False)
                accuracies_per_outcome_pre[outcome_value][gl.classifiers.index(model)].append(accuracy_results)
                accuracy_var_per_outcome_pre[outcome_value][gl.classifiers.index(model)].append(accuracy_var)
                f1_scores_per_outcome_pre[outcome_value][gl.classifiers.index(model)].append(f1_scores)
                f1_score_var_per_outcome_pre[outcome_value][gl.classifiers.index(model)].append(f1_var)

                # Train and predict with k-fold validation for POST
                accuracy_results, accuracy_var, f1_scores, f1_var, tmp0, tmp1 = clf.classify_k_fold(
                    post_marker_single_map,
                    outcome_value,
                    result_path, [],
                    model, False, False)
                accuracies_per_outcome_post[outcome_value][gl.classifiers.index(model)].append(accuracy_results)
                accuracy_var_per_outcome_post[outcome_value][gl.classifiers.index(model)].append(accuracy_var)
                f1_scores_per_outcome_post[outcome_value][gl.classifiers.index(model)].append(f1_scores)
                f1_score_var_per_outcome_post[outcome_value][gl.classifiers.index(model)].append(f1_var)

                # Remove feature again from trial set
                pre_marker_single_map.pop(pre_key)
                post_marker_single_map.pop(post_key)

        # Save the results to an Excel file
        for pre_key, post_key in zip(pre_marker_map.keys(), post_marker_map.keys()):

            # Skip outcomes
            if pre_key in gl.original_outcome_strings or post_key in gl.original_outcome_strings:
                continue

            # Extract current values of iteration from overall scores
            temp_accuracies = []
            for val in accuracies_per_outcome_pre[outcome_value]:
                temp_accuracies.append(val[list(pre_marker_map.keys()).index(pre_key) - gl.number_outcomes])
            temp_accuracy_var = []
            for val in accuracy_var_per_outcome_pre[outcome_value]:
                temp_accuracy_var.append(val[list(pre_marker_map.keys()).index(pre_key) - gl.number_outcomes])
            temp_f1_scores = []
            for val in f1_scores_per_outcome_pre[outcome_value]:
                temp_f1_scores.append(val[list(pre_marker_map.keys()).index(pre_key) - gl.number_outcomes])
            temp_f1_score_var = []
            for val in f1_score_var_per_outcome_pre[outcome_value]:
                temp_f1_score_var.append(val[list(pre_marker_map.keys()).index(pre_key) - gl.number_outcomes])
            fe.save_results_to_file(np.array(temp_accuracies).flatten(),
                                    np.array(temp_accuracy_var).flatten(),
                                    np.array(temp_f1_scores).flatten(),
                                    np.array(temp_f1_score_var).flatten(),
                                    result_file_path, pre_key)

            # Extract current values of iteration from overall scores
            temp_accuracies = []
            for val in accuracies_per_outcome_post[outcome_value]:
                temp_accuracies.append(val[list(post_marker_map.keys()).index(post_key) - gl.number_outcomes])
            temp_accuracy_var = []
            for val in accuracy_var_per_outcome_post[outcome_value]:
                temp_accuracy_var.append(val[list(post_marker_map.keys()).index(post_key) - gl.number_outcomes])
            temp_f1_scores = []
            for val in f1_scores_per_outcome_post[outcome_value]:
                temp_f1_scores.append(val[list(post_marker_map.keys()).index(post_key) - gl.number_outcomes])
            temp_f1_score_var = []
            for val in f1_score_var_per_outcome_post[outcome_value]:
                temp_f1_score_var.append(val[list(post_marker_map.keys()).index(post_key) - gl.number_outcomes])
            fe.save_results_to_file(np.array(temp_accuracies).flatten(),
                                    np.array(temp_accuracy_var).flatten(),
                                    np.array(temp_f1_scores).flatten(),
                                    np.array(temp_f1_score_var).flatten(),
                                    result_file_path, pre_key)


# Compute T-Test for performance comparison of each sensible subset combination
def t_test_to_different_subsets_performance(complete_data_map, result_path):
    # All subsets that should be compared
    combinations_to_test = [['PRE', 'POST'],
                            ['PRE', 'PMP'],
                            ['POST', 'PMP'],
                            ['PRE_POST', 'BEFORE_AFTER'],
                            ['PRE_POST', 'PRE_POST_BEFORE_AFTER'],
                            ['BEFORE_AFTER', 'PRE_POST_BEFORE_AFTER']]

    # Use 10-fold cross validation here
    gl.k_fold_split = 10

    # Do it for all outcomes and classifiers
    for outcome_index in range(gl.number_outcomes):
        for classification_descriptor in gl.classifiers:
            for combination in combinations_to_test:
                f1_results = []
                f1_means = []
                for data_set in combination:
                    gl.feature_blocks_to_use = data_set
                    # Get the F1-Score of the classification for the set-classifier-outcome combination
                    tmp0, tmp1, f1_mean, tmp4, tmp5, f1_scores = clf.classify_k_fold(complete_data_map,
                                                                                     outcome_index,
                                                                                     result_path,
                                                                                     [],
                                                                                     classification_descriptor,
                                                                                     False, False)
                    f1_results.append(f1_scores)
                    f1_means.append(f1_mean[0])

                # Compute the t-test values
                t, p = stats.ttest_rel(f1_results[0], f1_results[1], axis=0)

                # Output the t-test results
                print(f'CV T-Test result for {gl.outcome_descriptors[outcome_index]} '
                      f'with {classification_descriptor} and sets {combination}\n'
                      f'F1 set 0: {f1_means[0]}, F1 set 1: {f1_means[1]}, t: {t}, p: {p}\n')
