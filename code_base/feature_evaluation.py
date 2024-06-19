from statsmodels.stats.outliers_influence import variance_inflation_factor
import classification as clf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os
import explorational_data_analysis as exp
import global_variables as gl
import preprocess_data as pre
import parameter_evaluation as pe


# Save results to Excel file
def save_results_to_file(accuracies, f1_scores, result_path, removed_feature_descriptor):
    model_descriptors = [(desc + '_accuracy') for desc in gl.classifiers] + [(desc + '_f1_score') for desc in
                                                                             gl.classifiers]

    # If file to save results does not exist yet create it with a descriptor for the columns
    if not os.path.exists(result_path):
        df = pd.DataFrame({
            'Removed_Feature': model_descriptors,
            removed_feature_descriptor: np.concatenate((accuracies, f1_scores))
        })
    # Otherwise simply append the new data column
    else:
        df = pd.read_excel(result_path)
        df[removed_feature_descriptor] = np.concatenate((accuracies, f1_scores))

    # Write out new stuff
    df.to_excel(result_path, index=False)


# Plot the scores of the different classifiers after eliminating features one by one
def plot_feature_ablation_results(accuracies_per_model, f1_scores_per_model, removed_features, result_path,
                                  outcome_descriptor):
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    feature_counts = list(range(len(removed_features), 0, -1))

    # Plot a scatter plot of the data including a regression line
    for model in range(len(accuracies_per_model)):
        # ax.scatter(feature_counts, accuracies_per_model[model], label=classifiers[model]+'_accuracy')
        sns.regplot(x=feature_counts, y=f1_scores_per_model[model], scatter_kws={"color": gl.classifier_colors[model]},
                    line_kws={"color": gl.classifier_colors[model]}, order=6, label=gl.classifiers[model])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Feature ablation study plot for ' + outcome_descriptor)
    plt.xlabel('Number of features left')
    plt.ylabel('F1 score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(result_path)
    plt.close()


# Calculate the vif for each feature of the data set and return the on with the highest value
def check_feature_variance_inflation(train_data_map, result_path):
    marker_names, marker_data = [], []
    # Exclude outcomes from features
    for feature_name, feature_data in train_data_map.items():
        marker_names.append(feature_name)
        marker_data.append(feature_data)

    # Create a pandas DataFrame
    df = pd.DataFrame(np.transpose(marker_data), columns=marker_names)
    vif_df = pd.DataFrame()
    vif_df["feature"] = df.columns

    # Us statsmodels library to compute vifs
    vif_df["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    # vifs = vif_df["VIF"]

    # Get the highest value and return it
    worst_feature = vif_df.loc[vif_df['VIF'].idxmax()]
    return worst_feature


# Test
def read_csv_results(result_directory, mapmap):
    stats_file_path = os.path.join(result_directory, "ablation_study_vifs.txt")

    odd_index_entries = []
    even_index_entries = []

    with open(stats_file_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            odd_entries = [row[i] for i in range(len(row)) if i % 2 != 0]  # odd indices
            even_entries = [row[i] for i in range(len(row)) if i % 2 == 0]  # even indices

            odd_index_entries.append(odd_entries)
            even_index_entries.append(even_entries)

        for entry in even_index_entries:
            mapmap.pop(entry)

    return mapmap


# Check the features of a data set by their vif and eliminate the highest in each iteration
# Then train classifiers and evaluate their performance without the feature
def perform_feature_ablation_study_vif(complete_data_map, result_directory):
    removed_features = []
    os.makedirs(result_directory, exist_ok=True)
    result_file_name = "feature_ablation_study_vif"
    stats_file_path = os.path.join(result_directory, "ablation_study_vifs.txt")
    result_path = os.path.join(result_directory, result_file_name)
    outcome_result_paths = [result_path + '_' + key for key in gl.outcome_descriptors]
    # Prepare data structures that hold outcomes of study
    accuracies_per_outcome, f1_scores_per_outcome = pe.create_result_structure()
    # Reference map for computing vifs
    reference_map = complete_data_map.copy()
    reference_map, rest = pre.preprocess_data(reference_map, reference_map.copy(), 0,
                                              gl.standardize, gl.impute, gl.max_test_threshold, gl.oversample)
    # Get rid of the first column that is the outcome
    reference_map.pop(list(reference_map.keys())[0])
    # Save length of map for later use
    number_of_features = len(list(reference_map.keys()))
    #  !!! Temporary pop most of the map for testing !!!
    # tmp = list(reference_map.keys())
    # for key in list(complete_data_map.keys())[10:len(complete_data_map.keys()) - 1]:
    #     complete_data_map.pop(key)
    #     if key in tmp:
    #         reference_map.pop(key)

    # Open file to save results
    with open(stats_file_path, 'w') as stats_file:

        # Perform feature elimination and classification until only one feature is left
        for feature_count in range(0, len(complete_data_map.keys()) - gl.number_outcomes - 2):
            # Use Variance Inflation Factors for each feature and get highest as worst
            worst_feature = check_feature_variance_inflation(reference_map, result_path)
            # Eliminate feature
            complete_data_map.pop(worst_feature['feature'])
            reference_map.pop(worst_feature['feature'])
            removed_features.append(worst_feature['feature'])
            print(f'Remove from training data:', worst_feature['feature'], "with vif:", worst_feature['VIF'],
                  ", features left:", number_of_features)
            number_of_features -= 1
            # Stop calculation with vif of 1, here there is no point in continuing (outcome would be random)
            if worst_feature['VIF'] >= 1.01:
                continue

            # Then perform classification with all given models and evaluate the performance with f1 score
            for model in gl.classifiers:
                for outcome_value in range(gl.number_outcomes):
                    # Find matching configuration in precomputed data
                    current_configurations = next((value for key, value in gl.preprocess_parameters.items()
                                                   if gl.outcome_descriptors[outcome_value] in key
                                                   and model in key), None)
                    # Define preprocessing parameters based on former optimization
                    gl.standardize = current_configurations[0]
                    gl.impute = current_configurations[1]
                    gl.z_score_threshold = current_configurations[2]
                    gl.oversample_rate = current_configurations[3]
                    # Train and predict with k-fold validation
                    accuracy_results, f1_scores = clf.classify_k_fold(complete_data_map, outcome_value,
                                                                      result_path, [],
                                                                      model, False, False)
                    accuracies_per_outcome[outcome_value][gl.classifiers.index(model)].append(accuracy_results)
                    f1_scores_per_outcome[outcome_value][gl.classifiers.index(model)].append(f1_scores)

            # Save results
            for outcome in range(gl.number_outcomes):
                # Extract current values of iteration from overall scores
                temp_accuracies = []
                for val in accuracies_per_outcome[outcome]: temp_accuracies.append(val[feature_count])
                temp_f1_scores = []
                for val in f1_scores_per_outcome[outcome]: temp_f1_scores.append(val[feature_count])
                save_results_to_file(np.array(temp_accuracies).flatten(), np.array(temp_f1_scores).flatten(),
                                     outcome_result_paths[outcome] + '.xlsx',
                                     exp.clean_feature_name(worst_feature['feature']))
            stats_file.write(f"{worst_feature['feature']},{worst_feature['VIF']},")

    # Plot resulting f1 score curve for each outcome and model
    for outcome in range(gl.number_outcomes):
        temp_accuracies = accuracies_per_outcome[outcome]
        temp_f1_scores = f1_scores_per_outcome[outcome]
        plot_feature_ablation_results(temp_accuracies, temp_f1_scores, removed_features,
                                      outcome_result_paths[outcome] + '_plot',
                                      gl.outcome_descriptors[outcome])


# Check the features of a data set by their influence on performance and eliminate the worst in each iteration
# Therefore train classifiers and evaluate their performance without the feature
def perform_feature_ablation_study_performance(complete_data_map, result_directory):
    #  !!! Temporary kick out most data to quicken debugging !!!
    for kickout in range(len(complete_data_map.keys()) - 1, 50, -1):
        complete_data_map.pop(list(complete_data_map.keys())[kickout])
    # Prepare data structures
    os.makedirs(result_directory, exist_ok=True)
    result_file_name = "feature_ablation_study_performance"
    result_path = os.path.join(result_directory, result_file_name)
    outcome_result_paths = [result_path + '_' + key for key in gl.outcome_descriptors]

    # Do all iterations per outcome
    for outcome in range(gl.number_outcomes):

        removed_features, accuracies_per_model, f1_scores_per_model = [], [], []

        # Consider each model per outcome separately, performance according to feature selection may vary
        for model in range(len(gl.classifiers)):
            removed_features.append([])
            accuracies_per_model.append([])
            f1_scores_per_model.append([])
            reference_map = complete_data_map.copy()
            stats_file_path = os.path.join(result_directory,
                                           gl.outcome_descriptors[outcome] + '_'
                                           + gl.classifiers[model] + "_ablation_study_performance.txt")

            # Do loop over all features, classify and save f1
            # Open file to save results
            with open(stats_file_path, 'w') as stats_file:

                # Perform feature elimination and classification until only a few features are left
                for feature_count in range(0, len(complete_data_map.keys()) - gl.number_outcomes - 2):
                    removed_feature_trials = []
                    accuracy_ablation_results = []
                    f1_score_ablation_results = []

                    # Leave out each feature and train on remaining data
                    for feature_name, feature_data in reference_map.copy().items():
                        # Skip outcome classes
                        if feature_name in gl.original_outcome_strings or any(isinstance(elem, str) for elem in feature_data):
                            continue

                        # Delete feature testwise
                        reference_map.pop(feature_name)
                        removed_feature_trials.append(feature_name)
                        # Classify with the reduced set
                        accuracy_result, f1_score = clf.classify_k_fold(reference_map, outcome,
                                                                        result_path, [],
                                                                        gl.classifiers[model], False, False)
                        accuracy_ablation_results.append(accuracy_result)
                        f1_score_ablation_results.append(f1_score)
                        # Add feature back to set
                        reference_map[feature_name] = feature_data

                    # Extract feature whose deletion leads to the biggest performance gain
                    worst_feature_idx = f1_score_ablation_results.index(max(f1_score_ablation_results))
                    worst_feature = removed_feature_trials[worst_feature_idx]
                    # Save performance of classification after feature is deleted
                    accuracies_per_model[model].append(accuracy_ablation_results[worst_feature_idx])
                    f1_scores_per_model[model].append(f1_score_ablation_results[worst_feature_idx])
                    # Eliminate feature with worst result
                    reference_map.pop(worst_feature)
                    removed_features[model].append(worst_feature)
                    # Save information about what was deleted
                    stats_file.write(f"{worst_feature},{f1_score_ablation_results[worst_feature_idx]},")
                    print('Feature', worst_feature, 'deleted for', gl.outcome_descriptors[outcome],
                          gl.classifiers[model], 'with F1-Score', f1_score_ablation_results[worst_feature_idx])

        # Save plot that compares classifiers for each outcome
        plot_feature_ablation_results(accuracies_per_model, f1_scores_per_model, removed_features[0],
                                      outcome_result_paths[outcome] + '_plot',
                                      gl.outcome_descriptors[outcome])
