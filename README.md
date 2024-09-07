
# Stock Price Prediction using Linear Regression

This project involves predicting stock prices using historical data. The provided script utilizes a linear regression model to forecast future stock prices based on past data.

## Requirements

Ensure you have the following Python packages installed:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `keras`

You can install these packages using `pip`. Create a `requirements.txt` file with the following content:

pandas  
numpy  
matplotlib  
scikit-learn  
keras  

Then, install the requirements with:

pip install -r requirements.txt

## Script Description

The script performs the following steps:

1. **Load Data**: Reads historical stock price data from a CSV file.
2. **Data Preprocessing**:
   - Ensures column names are consistent.
   - Converts the `date` column to datetime format and sets it as the index.
   - Converts `close` and `volume` columns to float type.
3. **Feature and Target Creation**:
   - Defines features (`open`, `high`, `low`, `volume`) and target (`close`).
4. **Data Splitting**: Splits the data into training and test sets.
5. **Model Definition**: Creates a sequential neural network model with dense layers.
6. **Training**: Trains the model on the training data.
7. **Prediction**: Makes predictions on the test data.
8. **Evaluation**: Calculates and prints Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
9. **Visualization**: Plots the true values and predictions, then saves the plot as `forecast_plot.png`.

## Running the Script

To run the script, ensure you have the required packages installed and the CSV file (`aapl_us_d.csv`) available in the working directory. Execute the script with:

python script_name.py

Replace `script_name.py` with the name of your script file.

## Output

The script generates a file named `forecast_plot.png` showing the predicted stock prices versus the true values.

## Example Output

After running the script, you should see output similar to:

Linear Regression - MAE: [value], RMSE: [value]

A plot will be saved as `forecast_plot.png` displaying the comparison between true values and predictions.
