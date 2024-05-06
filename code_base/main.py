import read_excel
import explorational_data_analysis as exp
import correlation_analysis as cor
import preprocess_data as pre
import os


def main():
    # Path to Excel files with training and test data
    train_excel_file = 'data\\preprocessed_train_data.xlsx'
    test_excel_file = 'data\\preprocessed_test_data.xlsx'
    source_dir_path = os.getcwd()
    train_source_path = os.path.join(source_dir_path, train_excel_file)
    test_source_path = os.path.join(source_dir_path, test_excel_file)
    result_path = os.path.join(source_dir_path, "exploration_results")

    # Read data from Excel file
    train_data_map = read_excel.read_excel_data(train_source_path)
    test_data_map = read_excel.read_excel_data(test_source_path)

    # Preprocess data for exploration
    standardize, impute, filter_outliers = True, True, False
    pre.preprocess_data(train_data_map, [], standardize=standardize, impute=impute, filter_outliers=filter_outliers)
    pre.preprocess_data(test_data_map, train_data_map, standardize=standardize, impute=impute, filter_outliers=False)
    if standardize or impute:
        result_path = os.path.join(source_dir_path, "standardized_exploration_results")

    # Explore data and save results
    exp.check_data_sets(train_data_map, test_data_map)
    # exp.explore_data(train_data_map, result_path)

    # Compute correlation of markers and outcomes
    # cor.compute_marker_to_outcome_correlation(train_data_map, result_path)
    # cor.compute_marker_correlation_matrix(train_data_map, result_path)
    # cor.show_pairwise_marker_correlation(train_data_map, result_path)

    ''' 
    # Print the resulting dictionary

    for feature_name, feature_data in data_map.items():
        print(f"Feature Name: {feature_name}")
        print(f"Feature Data: {feature_data}")
        print()
    '''


if __name__ == "__main__":
    main()
