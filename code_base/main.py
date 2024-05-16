import read_excel
import explorational_data_analysis as exp
import correlation_analysis as cor
import preprocess_data as pre
import classification as clf
import feature_evaluation as fe
import parameter_evaluation as pe
import os

classifiers = ['NaiveBayes', 'LinearRegression', 'DecisionTree', 'SVM', 'RandomForest']


def main():
    # Path to Excel files with training and test data
    train_excel_file = 'data\\preprocessed_train_data.xlsx'
    test_excel_file = 'data\\preprocessed_test_data.xlsx'
    complete_excel_file = 'data\\preprocessed_complete_data.xlsx'
    source_dir_path = os.getcwd()
    train_source_path = os.path.join(source_dir_path, train_excel_file)
    test_source_path = os.path.join(source_dir_path, test_excel_file)
    complete_source_path = os.path.join(source_dir_path, complete_excel_file)
    result_path = os.path.join(source_dir_path, "results")

    # Read data from Excel file
    train_data_map = read_excel.read_excel_data(train_source_path)
    test_data_map = read_excel.read_excel_data(test_source_path)
    complete_data_map = read_excel.read_excel_data(complete_source_path)

    # Tune parameters
    # 1. Find best z-score outlier filter value
    parameter_evaluation_result_path = os.path.join(result_path, "parameter_evaluation_results")
    pe.find_best_z_score_filter(train_data_map, test_data_map, classifiers, parameter_evaluation_result_path)

    # Preprocess data for exploration and classification
    #   Standardization can be turned on and off
    #   For imputation a method can be chosen
    #   For outlier filtering a threshold can be chosen
    #   For validation you can choose 'hold_out' for standard 80/20 distribution,
    #       'k_fold' for k-set cross validation and 'leave_one_out' for n-point cross validation
    standardize, impute, filter_outliers_z_score, validation_method = True, True, 0, 'hold_out'
    pre.preprocess_data(train_data_map, standardize=standardize, impute=impute,
                        z_score_threshold=filter_outliers_z_score)
    pre.preprocess_data(test_data_map, standardize=standardize, impute=impute,
                        z_score_threshold=0)
    pre.preprocess_data(complete_data_map, standardize=standardize, impute=impute,
                        z_score_threshold=filter_outliers_z_score)

    # Explore data and save results
    # exp.check_data_sets(train_data_map, test_data_map)
    # if standardize or impute:
    #     exp.explore_data(train_data_map, os.path.join(result_path, "standardized_exploration_results"))
    # else:
    #     exp.explore_data(train_data_map, os.path.join(result_path, "exploration_results"))

    # Compute correlation metrics
    # cor.compute_marker_to_outcome_correlation(train_data_map, os.path.join(result_path, "correlation_results"))
    # cor.compute_marker_correlation_matrix(train_data_map, os.path.join(result_path, "correlation_results"))
    # cor.show_pairwise_marker_correlation(train_data_map, os.path.join(result_path, "correlation_results"))



    # Train classifier and predict
    classification_result_path = os.path.join(result_path, "classification_results")
    parameter_descriptor = [standardize, impute, filter_outliers_z_score]
    # Turn this on to retrieve model configuration
    print_model_details = False

    # if validation_method == 'hold_out':
    #     for model in classifiers:
    #         clf.classify(train_data_map, test_data_map, classification_result_path, parameter_descriptor,
    #                      model, print_model_details, True)

    # if validation_method == 'k_fold':
    #     for model in classifiers:
    #         clf.classify_k_fold(complete_data_map, classification_result_path, parameter_descriptor,
    #                             model, print_model_details, True)

    # Do an ablation study to eliminate features from data set
    feature_evaluation_result_path = os.path.join(result_path, "feature_evaluation_results")
    # fe.perform_ablation_study(train_data_map, test_data_map, feature_evaluation_result_path, classifiers)

    # accuracies_per_model = [[0.5, 0.3, 0.6], [0.8, 0.3, 0.6], [0.5, 0.3, 0.7], [0.8, 0.3, 0.7], [0.5, 0.3, 0.7]]
    # f1_scores_per_model = [[0.2, 0.2, 0.1], [0.1, 0.2, 0.1], [0.05, 0.2, 0.1], [0.2, 0.6, 0.1], [0.2, 0.8, 0.1]]
    # removed_features = ['1', '2', '3']
    # fe.plot_feature_ablation_results(accuracies_per_model, f1_scores_per_model, removed_features, result_path, classifiers, 'test')

    ''' 
    # Print the resulting dictionary

    for feature_name, feature_data in data_map.items():
        print(f"Feature Name: {feature_name}")
        print(f"Feature Data: {feature_data}")
        print()
    '''


if __name__ == "__main__":
    main()
