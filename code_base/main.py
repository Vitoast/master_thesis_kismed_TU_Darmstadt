import read_excel
import explorational_data_analysis as exp
import correlation_analysis as cor
import preprocess_data as pre
import classification as clf
import feature_evaluation as fe
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

    # Preprocess data for exploration and classification
    #   Standardization can be turned on and off
    #   For imputation a method can be chosen
    #   For outlier filtering a threshold can be chosen
    #   For validation you can choose 'hold_out' for standard 80/20 distribution,
    #       'k_fold' for k-set cross validation and 'leave_one_out' for n-point cross validation
    standardize, impute, filter_outliers, validation_method = True, True, False, 'hold_out'
    pre.preprocess_data(train_data_map, [], standardize=standardize, impute=impute,
                        filter_outliers=filter_outliers)
    pre.preprocess_data(test_data_map, train_data_map, standardize=standardize, impute=impute,
                        filter_outliers=False)
    pre.preprocess_data(complete_data_map, [], standardize=standardize, impute=impute,
                        filter_outliers=filter_outliers)

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
    parameter_descriptor = [standardize, impute, filter_outliers]
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
    fe.perform_ablation_study(train_source_path, test_data_map, feature_evaluation_result_path, classifiers)

    ''' 
    # Print the resulting dictionary

    for feature_name, feature_data in data_map.items():
        print(f"Feature Name: {feature_name}")
        print(f"Feature Data: {feature_data}")
        print()
    '''


if __name__ == "__main__":
    main()
