import read_excel
import explorational_data_analysis as exp
import correlation_analysis as cor
import os


def main():
    # Path to Excel file
    excel_file = 'data\\preprocessed_train_data.xlsx'
    source_dir_path = os.getcwd()
    source_path = os.path.join(source_dir_path, excel_file)
    result_path = os.path.join(source_dir_path, "exploration_results")

    # Read data from Excel file
    data_map = read_excel.read_excel_data(source_path)

    # Explore data and save results
    # exp.explore_data(data_map, result_path)

    # Compute correlation of markers and outcomes
    # cor.compute_marker_to_outcome_correlation(data_map, result_path)
    # cor.compute_marker_correlation_matrix(data_map, result_path)
    cor.show_pairwise_marker_correlation(data_map, result_path)

    ''' 
    # Print the resulting dictionary

    for feature_name, feature_data in data_map.items():
        print(f"Feature Name: {feature_name}")
        print(f"Feature Data: {feature_data}")
        print()
    '''


if __name__ == "__main__":
    main()
