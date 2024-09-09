import pandas as pd


# Read the Excel files that contain the data set and return them as a dictionary
def read_excel_data(file_path):
    # Read data from Excel file
    df = pd.read_excel(file_path, skiprows=0)
    data_dict = {}

    # Iterate over columns in the dataframe
    for column in df.columns:
        # Extract feature name from the second row
        feature_name = str(df[column][0])
        # Extract data from column, excluding the first row
        feature_data = df[column][1:].values.tolist()
        # Add feature name and data to dictionary
        data_dict[feature_name] = feature_data

    # Return dictionary that contains first row as keys and columns as values
    return data_dict
