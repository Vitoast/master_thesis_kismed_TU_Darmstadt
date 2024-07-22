from conda_build import post
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os

import global_variables as gl
import explorational_data_analysis as exp
import preprocess_data as pre
import parameter_evaluation as pe
import classification as clf


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
        if len(removed_features) > 50:
            sns.regplot(x=feature_counts, y=f1_scores_per_model[model],
                        scatter_kws={"color": gl.classifier_colors[model]},
                        line_kws={"color": gl.classifier_colors[model]}, order=6, label=gl.classifiers[model])
        else:
            plt.scatter(x=feature_counts, y=f1_scores_per_model[model], color=gl.classifier_colors[model],
                        label=gl.classifiers[model])
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

    # Get the highest value and return it
    worst_feature = vif_df.loc[vif_df['VIF'].idxmax()]
    return worst_feature


# Read in the VIF result file of a vif ablation study, there is one for all outcomes
# It has one line with each deleted feature and its vif comma separated
def read_feature_ablation_csv_file_vif(stats_file_path):
    marker_names = []
    f1_scores = []

    with open(stats_file_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # Divide data in markers and their VIFs
        for row in reader:
            current_vifs = [row[i] for i in range(len(row)) if i % 2 != 0]  # odd indices
            current_markers = [row[i] for i in range(len(row)) if i % 2 == 0]  # even indices

            marker_names.append(current_markers)
            f1_scores.append(current_vifs)

    return marker_names, f1_scores


# Read in the performance result file of a vif ablation study, there is one for each outcome
# It contains the accuracy and f1-scores for each model by each deleted feature
def read_feature_ablation_excel_file_per_outcome_vif(stats_file_path):
    all_f1_scores, all_accuracies = [], []

    # Read the Excel file
    df = pd.read_excel(stats_file_path)

    # Get the first row as names of the deleted features
    feature_names = df.columns[1:].tolist()

    # Extract data for each classifier that should be considered
    for classifier in gl.classifiers:
        accuracy_row = df[df['Removed_Feature'] == f"{classifier}_accuracy"]
        f1_score_row = df[df['Removed_Feature'] == f"{classifier}_f1_score"]
        if not accuracy_row.empty and not f1_score_row.empty:
            all_accuracies.append(accuracy_row.iloc[0, 1:].values)
            all_f1_scores.append(f1_score_row.iloc[0, 1:].values)

    return feature_names, all_f1_scores, all_accuracies


# Read the results from a former performance feature ablation study
# It has one line with the deleted feature, the f1-score and the accuracy of each iteration comma separated
def read_feature_ablation_csv_file_performance(filename):
    markers, f1_scores, accuracies = [], [], []

    # Append each point to the correct structure according to its position
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            for i, value in enumerate(row):
                if i % 3 == 0:
                    markers.append(value)
                elif i % 3 == 1:
                    f1_scores.append(float(value.strip('[]')))
                elif i % 3 == 2:
                    accuracies.append(float(value.strip('[]')))

    return markers, f1_scores, accuracies


# Check the features of a data set by their vif and eliminate the highest in each iteration
# Then train classifiers and evaluate their performance without the feature
def perform_feature_ablation_study_vif(original_data_map, result_directory):
    removed_features = []
    os.makedirs(result_directory, exist_ok=True)
    result_file_name = "feature_ablation_study_vif"
    stats_file_path = os.path.join(result_directory, "ablation_study_vifs.txt")
    leftover_features_file_path = os.path.join(result_directory, "leftover_features_vif.txt")
    result_path = os.path.join(result_directory, result_file_name)
    outcome_result_paths = [result_path + '_' + key for key in gl.outcome_descriptors]
    # Prepare data structures that hold outcomes of study
    accuracies_per_outcome, f1_scores_per_outcome = pe.create_result_structure()
    # Reference map for computing vifs
    complete_data_map = original_data_map.copy()
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

    stop_iterations = False
    # Open file to save results
    with open(stats_file_path, 'w') as stats_file:

        # Perform feature elimination and classification until only one feature is left
        for feature_count in range(0, len(complete_data_map.keys()) - gl.number_outcomes - 2):
            # Use Variance Inflation Factors for each feature and get highest as worst
            worst_feature = check_feature_variance_inflation(reference_map, result_path)
            # Stop calculation with vif of 1, here there is no point in continuing (outcome would be random)
            if worst_feature['VIF'] <= gl.vif_threshold:
                stop_iterations = True
                break
            # Eliminate feature
            complete_data_map.pop(worst_feature['feature'])
            reference_map.pop(worst_feature['feature'])
            removed_features.append(worst_feature['feature'])
            print(f'Remove from training data:', worst_feature['feature'], "with vif:", worst_feature['VIF'],
                  ", features left:", number_of_features)
            number_of_features -= 1

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

    # Save leftover features
    if stop_iterations:
        with open(leftover_features_file_path, 'w') as leftover_features:
            for feature_name in reference_map.keys():
                leftover_features.write(f"{feature_name},")

    # Plot resulting f1 score curve for each outcome and model
    for outcome in range(gl.number_outcomes):
        temp_accuracies = accuracies_per_outcome[outcome]
        temp_f1_scores = f1_scores_per_outcome[outcome]
        plot_feature_ablation_results(temp_accuracies, temp_f1_scores, removed_features,
                                      outcome_result_paths[outcome] + '_plot',
                                      gl.outcome_descriptors[outcome])


# To continue an ablation study with vifs of only 1.0 performance can be used as measure
# Since it would be random otherwise
def continue_performance_ablation_after_vif(result_directory, result_file, complete_data_map):
    os.makedirs(result_directory, exist_ok=True)
    if result_file == "":
        result_file = os.path.join(result_directory, "ablation_study_vifs.txt")
    # Read vif ablation results
    deleted_features, vifs = read_feature_ablation_csv_file_vif(result_file)
    # Remove deleted features from set
    working_map = complete_data_map.copy()
    for feature_name in sum(deleted_features, []):
        if feature_name == "": continue
        working_map.pop(feature_name)
    # Continue ablation with remaining features
    perform_feature_ablation_study_performance(working_map, result_directory)


# Check the features of a data set by their influence on performance and eliminate the worst in each iteration
# Therefore train classifiers and evaluate their performance without the feature
def perform_feature_ablation_study_performance(complete_data_map, result_directory):
    #  !!! Temporary kick out most data to quicken debugging !!!
    # for kickout in range(len(complete_data_map.keys()) - 1, 10, -1):
    #     complete_data_map.pop(list(complete_data_map.keys())[kickout])
    # Prepare data structures
    os.makedirs(result_directory, exist_ok=True)
    result_file_name = "feature_ablation_study_performance"
    result_path = os.path.join(result_directory, result_file_name)
    outcome_result_paths = [result_path + '_' + key for key in gl.outcome_descriptors]

    complete_data_map = pre.filter_data_sub_sets(complete_data_map)

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
                for feature_count in range(0, len(complete_data_map.keys()) - gl.number_outcomes):
                    removed_feature_trials = []
                    accuracy_ablation_results = []
                    f1_score_ablation_results = []

                    # Leave out each feature and train on remaining data
                    for feature_name, feature_data in reference_map.copy().items():
                        # Skip outcome classes
                        if feature_name in gl.original_outcome_strings or any(
                                isinstance(elem, str) for elem in feature_data):
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
                    stats_file.write(
                        f"{worst_feature},{f1_score_ablation_results[worst_feature_idx]},{accuracy_ablation_results[worst_feature_idx]},")
                    print('Feature', worst_feature, 'deleted for', gl.outcome_descriptors[outcome],
                          gl.classifiers[model], 'with F1-Score', f1_score_ablation_results[worst_feature_idx])

        # Save plot that compares classifiers for each outcome
        plot_feature_ablation_results(accuracies_per_model, f1_scores_per_model, removed_features[0],
                                      outcome_result_paths[outcome] + '_plot',
                                      gl.outcome_descriptors[outcome])


# Read the saved SCV file from a performance ablation study and plot the results
def plot_former_feature_ablation(result_directory):
    all_f1_scores, all_accuracies = pe.create_result_structure()
    os.makedirs(result_directory, exist_ok=True)
    result_file_name = "feature_ablation_study_performance"
    result_path = os.path.join(result_directory, result_file_name)
    outcome_result_paths = [result_path + '_' + key for key in gl.outcome_descriptors]

    # Create a plot for each outcome
    for outcome in range(gl.number_outcomes):
        markers = []
        # Load results for each model
        for model in range(len(gl.classifiers)):
            stats_file_path = os.path.join(result_directory,
                                           gl.outcome_descriptors[outcome] + '_'
                                           + gl.classifiers[model] + "_ablation_study_performance.txt")
            markers, f1_scores, accuracies = read_feature_ablation_csv_file_performance(stats_file_path)
            all_f1_scores[outcome][model] = f1_scores
            all_accuracies[outcome][model] = accuracies

        # Remove last point if it is a "" from the end of the file
        if "" in markers: markers.pop(len(markers) - 1)

        plot_feature_ablation_results(all_accuracies[outcome], all_f1_scores[outcome], markers,
                                      outcome_result_paths[outcome] + '_replot', gl.outcome_descriptors[outcome])


# In case of combined feature ablation of VIF and performance use this to plot the results of each model
def plot_one_model_vif_and_performance_feature_ablation(result_directory, only_performance=False):
    all_f1_scores, all_accuracies = pe.create_result_structure()
    plot_save_directory = os.path.join(result_directory, 'plots_of_combined_ablation')
    os.makedirs(plot_save_directory, exist_ok=True)

    # Plot for each model separately
    for model in gl.classifiers:
        # Get results for each outcome from saved files
        for outcome in range(gl.number_outcomes):

            all_markers_in_order, all_f1_scores[outcome], all_accuracies[outcome] = [], [], []
            for feature_string in gl.classifiers:
                all_f1_scores[outcome].append([])
                all_accuracies[outcome].append([])

            if not only_performance:
                vif_file_name = 'feature_ablation_study_vif_' + gl.outcome_descriptors[outcome] + '.xlsx'
                vif_file_path = os.path.join(result_directory, vif_file_name)

                # First get the results of the VIF ablation study
                markers, f1_scores, accuracies = read_feature_ablation_excel_file_per_outcome_vif(vif_file_path)
                all_markers_in_order = markers
                all_f1_scores[outcome] = f1_scores
                all_accuracies[outcome] = accuracies

            performance_file_name = gl.outcome_descriptors[outcome] + '_' + model + '_ablation_study_performance.txt'
            performance_file_path = os.path.join(result_directory, performance_file_name)

            # Second get the results of the following performance ablation study
            markers, f1_scores, accuracies = read_feature_ablation_csv_file_performance(performance_file_path)
            vif_border = len(markers)
            # Delete "" point at the end
            if "" in markers: markers.pop(len(markers) - 1)
            all_markers_in_order = all_markers_in_order + markers
            # Add results to each model
            for i in range(len(all_f1_scores[outcome])):
                if not only_performance:
                    all_f1_scores[outcome][i] = list(np.concatenate((all_f1_scores[outcome][i], np.array(f1_scores))))
                    all_accuracies[outcome][i] = list(
                        np.concatenate((all_accuracies[outcome][i], np.array(accuracies))))
                else:
                    all_f1_scores[outcome][i] = f1_scores
                    all_accuracies[outcome][i] = list(
                        np.concatenate((all_accuracies[outcome][i], np.array(accuracies))))

        plot_save_name = os.path.join(plot_save_directory, 'combined_ablation_plot_' + model + '.png')

        # Start plotting of the results
        plt.figure(figsize=(12, 6))
        ax = plt.subplot(111)
        feature_counts = list(range(len(all_markers_in_order), 0, -1))

        # Plot a scatter plot of the data
        for outcome in range(gl.number_outcomes):
            plt.scatter(x=feature_counts, y=all_f1_scores[outcome][gl.classifiers.index(model)],
                        color=gl.classifier_colors[outcome],
                        label=gl.outcome_descriptors[outcome])
        box = ax.get_position()
        # Add vertical line to represent transition between VIF and performance
        if not only_performance:
            plt.axvline(x=vif_border, color='red', linestyle='--', linewidth=2,
                        label='Threshold between VIF\nand performance ablation')
            plt.title('Combined VIF and performance feature ablation study plot for ' + model)
        else:
            plt.title('Performance feature ablation study plot for ' + model)
        ax.set_position([box.x0, box.y0, box.width * 1, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('Number of features left')
        plt.ylabel('F1 score')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_save_name)
        plt.close()


# Perform a bottom up accumulation feature analysis
# Start with training the models with every single feature, select the best performing and add them one by one
# Additionally order each feature after their single performance and plot result like correlation
def perform_feature_accumulation(complete_data_map, result_directory):
    # Prepare data structures
    os.makedirs(result_directory, exist_ok=True)
    result_file_name = "feature_accumulation_study"
    result_path = os.path.join(result_directory, result_file_name)
    outcome_result_paths = [result_path + '_' + key for key in gl.outcome_descriptors]

    complete_data_map = pre.filter_data_sub_sets(complete_data_map)

    # Do all iterations per outcome
    for outcome in range(gl.number_outcomes):

        added_features, accuracies_per_model, f1_scores_per_model = [], [], []

        # Consider each model per outcome separately, performance according to feature selection may vary
        for model in range(len(gl.classifiers)):
            added_features.append([])
            accuracies_per_model.append([])
            f1_scores_per_model.append([])
            # Map to hold all features that are left
            reference_map = complete_data_map.copy()
            reduced_map = {}
            stats_file_path = os.path.join(result_directory,
                                           gl.outcome_descriptors[outcome] + '_'
                                           + gl.classifiers[model] + "_accumulation_study.txt")

            # Do loop over all features, classify and save f1
            # Open file to save results
            with open(stats_file_path, 'w') as stats_file:

                # Perform feature accumulation until total of 10 features,
                # above there was no performance gain in earlier evaluations
                for feature_count in range(0, 10):
                    added_feature_trials = []
                    accuracy_ablation_results = []
                    f1_score_ablation_results = []

                    # Add each feature and train on with resulting data set
                    for feature_name, feature_data in reference_map.items():
                        # Add feature to reference test map
                        reduced_map[feature_name] = feature_data

                        # Skip outcome classes
                        if feature_name in gl.original_outcome_strings or any(
                                isinstance(elem, str) for elem in feature_data):
                            continue

                        # Add feature testwise
                        added_feature_trials.append(feature_name)

                        # Classify with the reduced set
                        accuracy_result, f1_score = clf.classify_k_fold(reduced_map, outcome,
                                                                        result_path, [],
                                                                        gl.classifiers[model], False, False)
                        accuracy_ablation_results.append(accuracy_result[0])
                        f1_score_ablation_results.append(f1_score[0])

                        # Remove feature from set
                        reduced_map.pop(feature_name)

                    # Extract feature whose addition leads to the biggest performance gain
                    best_feature_idx = f1_score_ablation_results.index(max(f1_score_ablation_results))
                    best_feature = added_feature_trials[best_feature_idx]
                    # Save performance of classification after feature is added
                    accuracies_per_model[model].append(accuracy_ablation_results[best_feature_idx])
                    f1_scores_per_model[model].append(f1_score_ablation_results[best_feature_idx])
                    # Remove chosen feature from reference set
                    reference_map.pop(best_feature)
                    # Add it permanently to the trial set
                    reduced_map[added_feature_trials[best_feature_idx]] = complete_data_map[added_feature_trials[best_feature_idx]]
                    added_features[model].append(best_feature)
                    stats_file.write(f"{best_feature},"
                                     f"{f1_score_ablation_results[best_feature_idx]},"
                                     f"{accuracy_ablation_results[best_feature_idx]},\n")
                    print('Feature', best_feature, 'added for', gl.outcome_descriptors[outcome],
                          gl.classifiers[model], 'with F1-Score', f1_score_ablation_results[best_feature_idx])

                    # In the first round plot the f1-scores of all the features from their single prediction
                    if feature_count == 0:
                        # Sort the f1-scores descending and reorder the accuracies accordingly
                        sorted_indices = np.argsort(f1_score_ablation_results)[::-1]
                        f1_score_ablation_results = [f1_score_ablation_results[i] for i in sorted_indices]
                        accuracy_ablation_results = [accuracy_ablation_results[i] for i in sorted_indices]
                        marker_names_sorted = [added_feature_trials[i] for i in sorted_indices]

                        # Generate unique colors
                        colors = [plt.cm.get_cmap('viridis', len(marker_names_sorted))(i)
                                  for i in range(len(f1_score_ablation_results))]

                        # Create the subplots, size depending on the number of features/ length of the legend
                        if len(marker_names_sorted) > 101:
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))
                        else:
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

                        # for descriptor in ['Correlation Coefficient', 'P-Value']:
                        for i, (f1_score, color, name) in enumerate(
                                zip(f1_score_ablation_results, colors, marker_names_sorted)):
                            ax1.scatter(i, f1_score, color=color, label=name)
                        ax1.set_title(
                            'Sorted single feature performance study ' + gl.outcome_descriptors[outcome]
                            + ' using ' + gl.classifiers[model])
                        ax1.set_xlabel('Marker')
                        ax1.set_ylabel('F1-Score')
                        ax1.grid(True)

                        # Scatter plot for sorted p-values
                        for i, (acc, color) in enumerate(zip(accuracy_ablation_results, colors)):
                            ax2.scatter(i, acc, color=color)
                        ax2.set_title('Accuracies Corresponding to Sorted Features')
                        ax2.set_xlabel('Marker')
                        ax2.set_ylabel('Accuracy')
                        ax2.grid(True)

                        # Add legend besides plots
                        fig.tight_layout()
                        handles, labels = ax1.get_legend_handles_labels()
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
                        plt.legend(handles, labels, bbox_to_anchor=(1.1, 2.2), loc='upper left', fontsize='small',
                                   ncol=n_col)

                        # Save plot as PNG file
                        output_file_path = os.path.join(result_directory,
                                                        f'single_feature_performance_study_'
                                                        f'{gl.outcome_descriptors[outcome]}_{gl.classifiers[model]}.jpg')
                        plt.savefig(output_file_path)
                        plt.close()
                        # Save F1-Scores and accuracies for each single feature

                        single_results_file_path = os.path.join(result_directory,
                                                                gl.outcome_descriptors[outcome] + '_'
                                                                + gl.classifiers[model] + "_single_performance.txt")
                        with open(single_results_file_path, 'w') as res_file:
                            for ordered_feature in range(len(marker_names_sorted)):
                                res_file.write(f"{marker_names_sorted[ordered_feature]},"
                                               f"{f1_score_ablation_results[ordered_feature]},"
                                               f"{accuracy_ablation_results[ordered_feature]},\n")

        # Save plot that compares classifiers for each outcome
        plot_feature_ablation_results(accuracies_per_model, f1_scores_per_model, added_features[0],
                                      outcome_result_paths[outcome] + '_plot',
                                      gl.outcome_descriptors[outcome])


# Analyse the difference between the PRE and POST markers
# Models are trained with each single feature and the performance for both sets is saved in an Excel file per outcome
def compare_pre_to_post_marker_performance(complete_data_map, result_path):
    os.makedirs(result_path, exist_ok=True)
    result_file_path = os.path.join(result_path, 'pre_post_single_performance.xlsx')

    # Prepare data structures that hold outcomes of study
    accuracies_per_outcome_pre, f1_scores_per_outcome_pre = pe.create_result_structure()
    accuracies_per_outcome_post, f1_scores_per_outcome_post = pe.create_result_structure()

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
                accuracy_results, f1_scores = clf.classify_k_fold(pre_marker_single_map, outcome_value,
                                                                  result_path, [],
                                                                  model, False, False)
                accuracies_per_outcome_pre[outcome_value][gl.classifiers.index(model)].append(accuracy_results)
                f1_scores_per_outcome_pre[outcome_value][gl.classifiers.index(model)].append(f1_scores)

                # Train and predict with k-fold validation for POST
                accuracy_results, f1_scores = clf.classify_k_fold(post_marker_single_map, outcome_value,
                                                                  result_path, [],
                                                                  model, False, False)
                accuracies_per_outcome_post[outcome_value][gl.classifiers.index(model)].append(accuracy_results)
                f1_scores_per_outcome_post[outcome_value][gl.classifiers.index(model)].append(f1_scores)

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
                temp_accuracies.append(val[list(pre_marker_map.keys()).index(pre_key)-gl.number_outcomes])
            temp_f1_scores = []
            for val in f1_scores_per_outcome_pre[outcome_value]:
                temp_f1_scores.append(val[list(pre_marker_map.keys()).index(pre_key)-gl.number_outcomes])
            save_results_to_file(np.array(temp_accuracies).flatten(), np.array(temp_f1_scores).flatten(),
                                 result_file_path, pre_key)

            # Extract current values of iteration from overall scores
            temp_accuracies = []
            for val in accuracies_per_outcome_post[outcome_value]:
                temp_accuracies.append(val[list(post_marker_map.keys()).index(post_key)-gl.number_outcomes])
            temp_f1_scores = []
            for val in f1_scores_per_outcome_post[outcome_value]:
                temp_f1_scores.append(val[list(post_marker_map.keys()).index(post_key)-gl.number_outcomes])
            save_results_to_file(np.array(temp_accuracies).flatten(), np.array(temp_f1_scores).flatten(),
                                 result_file_path, post_key)
