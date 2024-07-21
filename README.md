# California House Price Prediction

Exploring data science projects!

## Brief Description
- **`requirements.txt`**: Contains all (some aren't necessary) the libraries to be installed in Anaconda environment. [I used python 3.12.0]
- **`1553768847-housing.csv`**: The dataset file.
- **`download_csv.py`**: Contains Python script to download the CSV file from Kaggle.
- **`model_metrics.py`**: Contains loaded dataset, models and metrices.
- **`app.py`**: Contains Streamlit app to predict results based on user inputs for user the selected model.

## Files' Contents

### download_csv.py
- Run this file to download the `.csv` directly from Kaggle.
- Before runing the scripts ensure that the `kaggle.json` file is in `c:\\Users\\admin\\.kaggle\\`.
- Ensure that the `kaggle.json` file has the content mentioned below,
```sh
{"username":"your_kaggle_username","key":"your_kaggle_key"}
```
[Make sure to replace "your_kaggle_username" and "your_kaggle_key" with your actual usernaem and kaggle key]

#### Alternative:
Run this command in cmd to download `dataset.zip` directly from Kaggle:
```sh
kaggle datasets download -d shibumohapatra/house-price
```

### 1553768847-housing.csv
#### Dataset File
- **Dataset**: `shibumohapatra/house-price`
- **File Name**: `1553768847-housing.csv`
- **Details**: Created after running `download_csv.py`, this file contains 20,640 rows and the following features: `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity`, `median_house_value`.

### requirements.txt
- Contains required packages/libraries to run all `.py` and `.ipynb` files.

### test.ipynb
- Contains test results of the house value prediction model.
- **ML Models used**: Linear Regression, Decision Tree, Random Forest, Gradient Boosting.

### model_metrics.py
- Contains functions to calculate model metrics, clean, encode, and scale the dataset. It also trains models on the dataset and evaluates their performance using RMSE, MAE, and R² scores.

### app.py
- Contains the Streamlit app.
- Use the command below in your cmd to run the "California House Price Prediction" app:
  ```sh
  streamlit run app.py
  ```