from statsmodels.stats.outliers_influence import variance_inflation_factor
import explorational_data_analysis as exp
import classification as clf
import pandas as pd
import numpy as np


number_outcomes = 5


def check_feature_variance_inflation(train_data_map, result_path):
    marker_names, marker_data = [], []
    feature_count = 0
    for feature_name, feature_data in train_data_map.items():
        if feature_count >= number_outcomes:
            marker_names.append(feature_name)
            marker_data.append(feature_data)
        feature_count += 1
    # Create a pandas DataFrame
    df = pd.DataFrame(np.transpose(marker_data), columns=marker_names)
    vif_df = pd.DataFrame()
    vif_df["feature"] = df.columns
    vif_df["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    vifs = vif_df["VIF"]
    worst_feature = vif_df.loc[vif_df['VIF'].idxmax()]
    return worst_feature, vifs


def perform_ablation_study(train_data_map, test_data_map, result_path, classifiers):
    removed_features = []
    accuracies_per_model, f1_scores_per_model = np.empty(len(classifiers)), np.empty(len(classifiers))
    for feature in train_data_map.keys():
        # Compute Variance Inflation Factors for each feature and remove the highest from the data set
        worst_feature, vifs = check_feature_variance_inflation(train_data_map, result_path)
        print(f'Remove from training data', worst_feature['feature'])
        train_data_map.pop(worst_feature['feature'])
        test_data_map.pop(worst_feature['feature'])
        removed_features.append(worst_feature['feature'])
        # Then perform classification with all models and evaluate the performance
        for model in classifiers:
            accuracy_results, f1_scores = clf.classify(train_data_map, test_data_map, result_path, [],
                                                       model, False, False)
            accuracies_per_model[classifiers.index(model)].append(accuracy_results)
            f1_scores_per_model[classifiers.index(model)].append(f1_scores)
            # Save and plot results
