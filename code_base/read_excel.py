import pandas as pd


def read_excel_data(file_path):
    # Read data from Excel file, skipping the first row
    df = pd.read_excel(file_path, skiprows=0)

    # Initialize dictionary to store data
    data_dict = {}

    # Iterate over columns in the dataframe
    for column in df.columns:
        # Extract feature name from the second row
        feature_name = str(df[column][0])

        # Extract data from column, excluding the first row
        feature_data = df[column][1:].values.tolist()

        # Add feature name and data to dictionary
        data_dict[feature_name] = feature_data

    return data_dict
