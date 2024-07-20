# datascience1 - Caliornia House Price
Exploring datascience projects!

## Contents
### download_csv.py
-Run the above file to download the .csv directly from kaggle [Make sure that kaggle.json file is in "c:\\Users\\admin\\.kaggle\\"]

#### or run this in cmd to download dataset.zip directly from kaggle
kaggle datasets download -d shibumohapatra/house-price 

### 1553768847-housing.csv
#### dataset-file
dataset = 'shibumohapatra/house-price'  # 'dataset-owner/dataset-name'
file_name = '1553768847-housing.csv'

-File created after running download_csv.py
-File contains 20640 rows
-Features: longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,ocean_proximity,median_house_value

### requirements.txt
-Contains required packages/libraries to run all .py and .ipynb files

### test.ipynb
-Contains test results of house_value prediction model
-ML models used: Linear Regression, Decision Tree, Random Forest, Gradient Boosting 
