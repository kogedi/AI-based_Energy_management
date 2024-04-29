import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read CSV file
def read_csv(filename):
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

# Calculate correlation matrix
def calculate_correlation_matrix(df):
    try:
        corr_matrix = df.corr()
        return corr_matrix
    except Exception as e:
        print(f"Error calculating correlation matrix: {e}")
        return None

def read_csv_from_folder(folder_path, file_name):
    import os
    
    dfs = {}
    #file_name = 'first_tier_data_set.csv'  # Update with your file name
    file_path = os.path.join(folder_path, file_name)
    try:
        df = pd.read_csv(file_path)
        dfs[file_name] = df
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
    return dfs

from datetime import datetime

def calculate_time_difference(start_timestamp, end_timestamp):
    try:
        # Convert string timestamps to datetime objects
        start_time = datetime.strptime(start_timestamp, '%Y-%m-%d %H:%M:%S.%f %Z')
        end_time = datetime.strptime(end_timestamp, '%Y-%m-%d %H:%M:%S.%f %Z')

        # Calculate time difference in seconds
        time_difference = (end_time - start_time).total_seconds()
        return time_difference
    except Exception as e:
        print(f"Error calculating time difference: {e}")
        return None


# Main function
def main():
    # Input CSV filename
    import os
    
    filename = 'first_tier_data_set.csv'
    folder_path = "C:/Users/Konra/git_repos/Cheftreff_1KOMMA5Grad"
    file_path = os.path.join(folder_path, 'data', filename)
    
    # Read CSV file
    df = read_csv(file_path)
    if df is None:
        return

    # Calculate time difference
    time_difference = np.zeros(df.shape[0])
    for index, row in df.iterrows():
        start_timestamp = row['start_time'] #'2023-10-10 15:18:30.000000 UTC'
        end_timestamp =  row['end_time'] #'2023-10-10 15:19:00.000000 UTC'
        time_difference[index] = calculate_time_difference(start_timestamp, end_timestamp)
    df.insert(loc=df.columns.get_loc('end_time') + 1, column='time_difference_sec', value=time_difference)
    # Calculate time difference
    # if time_difference is not None:
    #     print(f"Time difference: {time_difference} seconds")
    

    # Eliminate row start_time and row end_time from df
    df_filtered = df.drop(['start_time', 'end_time'], axis=1)

    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(df_filtered)
    if corr_matrix is None:
        return

    # Output correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()
    
if __name__ == "__main__":
    main()