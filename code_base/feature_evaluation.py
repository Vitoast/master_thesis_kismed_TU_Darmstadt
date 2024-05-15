from statsmodels.stats.outliers_influence import variance_inflation_factor
import classification as clf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

number_outcomes = 5
# Values to print names of adverse outcomes
outcome_descriptors = ["AKI3", "AKD1", "LCOS", "AF", "Any"]


# Save results to Excel file
def save_results_to_file(classifiers, accuracies, f1_scores, result_path, removed_feature_descriptor):
    model_descriptors = [(desc + '_accuracy') for desc in classifiers] + [(desc + '_f1_score') for desc in classifiers]
    if not os.path.exists(result_path):
        df = pd.DataFrame({
            'Removed_Feature': model_descriptors,
            removed_feature_descriptor: (accuracies + f1_scores)
        })
    else:
        df = pd.read_excel(result_path)
        df[removed_feature_descriptor] = (accuracies + f1_scores)
    df.to_excel(result_path, index=False)


# Plot the scores of the different classifiers after eliminating features one by one
def plot_feature_ablation_results(accuracies_per_model, f1_scores_per_model, removed_features, result_path, classifiers,
                                  outcome_descriptor):
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    feature_counts = list(range(len(removed_features), 0, -1))
    colors = ['red', 'orange', 'green', 'purple', 'black']
    # Plot a scatter plot of the data including a regression line
    for model in range(len(accuracies_per_model)):
        # ax.scatter(feature_counts, accuracies_per_model[model], label=classifiers[model]+'_accuracy')
        sns.regplot(x=feature_counts, y=f1_scores_per_model[model], scatter_kws={"color": colors[model]},
                    line_kws={"color": colors[model]}, order=6, label=classifiers[model])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Feature ablation study plot for ' + outcome_descriptor)
    plt.xlabel('Number of features left')
    plt.ylabel('F1 score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(result_path)
    plt.show()
    plt.close()


# Calculate the vif for each feature of the data set and return the on with the highest value
def check_feature_variance_inflation(train_data_map, result_path):
    marker_names, marker_data = [], []
    feature_count = 0
    # Exclude outcomes from features
    for feature_name, feature_data in train_data_map.items():
        if feature_count >= number_outcomes:
            marker_names.append(feature_name)
            marker_data.append(feature_data)
        feature_count += 1
    # Create a pandas DataFrame
    df = pd.DataFrame(np.transpose(marker_data), columns=marker_names)
    vif_df = pd.DataFrame()
    vif_df["feature"] = df.columns
    # Us statsmodels library to compute vifs
    vif_df["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    vifs = vif_df["VIF"]
    # Get highest value
    worst_feature = vif_df.loc[vif_df['VIF'].idxmax()]
    return worst_feature, vifs


# Check the features of a data set by their vif and eliminate the highest in each iteration
# Then train classifiers and evaluate their performance without the feature
def perform_ablation_study(train_data_map, test_data_map, result_directory, classifiers):
    removed_features = []
    accuracies_per_outcome, f1_scores_per_outcome = [], []
    os.makedirs(result_directory, exist_ok=True)
    result_file_name = "feature_ablation_study"
    stats_file_path = os.path.join(result_directory, "ablation_study_vifs.txt")
    result_path = os.path.join(result_directory, result_file_name)
    outcome_result_paths = [result_path + '_' + key for key in outcome_descriptors]
    # Prepare data structures that hold outcomes of study
    for i in range(number_outcomes):
        accuracies_per_outcome.append([])
        f1_scores_per_outcome.append([])
        for j in range(len(classifiers)):
            accuracies_per_outcome[i].append([])
            f1_scores_per_outcome[i].append([])
    # Open file to save results
    with open(stats_file_path, 'w') as stats_file:
        # Perform feature elimination and classification until only a few features are left
        for feature_count in range(0, len(train_data_map.keys()) - 10):
            # Compute Variance Inflation Factors for each feature and get highest
            worst_feature, vifs = check_feature_variance_inflation(train_data_map, result_path)
            print(f'Remove from training data:', worst_feature['feature'], " with vif:", worst_feature['VIF'], ", features left:", len(train_data_map) -1)
            # Eliminate feature
            train_data_map.pop(worst_feature['feature'])
            test_data_map.pop(worst_feature['feature'])
            removed_features.append(worst_feature['feature'])
            accuracy_results = []
            f1_scores = []
            # Then perform classification with all given models and evaluate the performance with f1 score
            for model in classifiers:
                accuracy_results, f1_scores = clf.classify(train_data_map, test_data_map, result_path, [],
                                                           model, False, False)
                for outcome_value in range(number_outcomes):
                    accuracies_per_outcome[outcome_value][classifiers.index(model)].append(accuracy_results[outcome_value])
                    f1_scores_per_outcome[outcome_value][classifiers.index(model)].append(f1_scores[outcome_value])
            # Save results
            for outcome in range(number_outcomes):
                save_results_to_file(classifiers, accuracy_results, f1_scores,
                                     outcome_result_paths[outcome] + '.xlsx', worst_feature['feature'])
            stats_file.write(f"{worst_feature['feature']},{worst_feature['VIF']},")
    # Plot resulting f1 score curve for each outcome and model
    for outcome in range(number_outcomes):
        temp_accuracies = accuracies_per_outcome[outcome]
        temp_f1_scores = f1_scores_per_outcome[outcome]
        plot_feature_ablation_results(temp_accuracies, temp_f1_scores, removed_features,
                                      outcome_result_paths[outcome] + '_plot', classifiers,
                                      outcome_descriptors[outcome])
