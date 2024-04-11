import read_excel
import os


def main():
    # Path to Excel file
    excel_file = 'DB_BRS-GR2013_Anonimous_CV_train.xlsx'
    source_path = os.path.join('..\\', excel_file)

    # Read data from Excel file
    data_map = read_excel.read_excel_data(source_path)

    # Print the resulting dictionary
    for feature_name, feature_data in data_map.items():
        print(f"Feature Name: {feature_name}")
        print(f"Feature Data: {feature_data}")
        print()


if __name__ == "__main__":
    main()
