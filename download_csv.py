import os
import kaggle
import pandas as pd
import zipfile

# Ensure the kaggle.json file is in the ~/.kaggle directory or set the environment variables directly
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')

# Replace with your Kaggle dataset details
dataset = 'shibumohapatra/house-price' ## author/projet
file_name = '1553768847-housing.csv' ##file-name

# Download the dataset file
kaggle.api.dataset_download_file(dataset, file_name, path='./')

# If the file is downloaded as a zip file, extract it
zip_file_path = f'{file_name}.zip'
if os.path.exists(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    # Delete the ZIP file after extraction
    os.remove(zip_file_path)

# Read the CSV file using pandas
data = pd.read_csv(file_name)
print(data.head())