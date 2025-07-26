import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import streamlit as st

def plot_regression(train_file, test_file):
    # Load the training CSV file
    train_df = pd.read_csv(train_file)

    # Split the 'Output0;Target0;Error0' column into separate columns
    train_df[['Output0', 'Target0', 'Error0']] = train_df['Output0;Target0;Error0'].str.split(';', expand=True)

    # Convert to numeric
    train_df['Output0'] = pd.to_numeric(train_df['Output0'])
    train_df['Target0'] = pd.to_numeric(train_df['Target0'])

    # Load the testing CSV file
    test_df = pd.read_csv(test_file)

    # Split the 'Output0;Target0;Error0' column into separate columns
    test_df[['Output0', 'Target0', 'Error0']] = test_df['Output0;Target0;Error0'].str.split(';', expand=True)

    # Convert to numeric
    test_df['Output0'] = pd.to_numeric(test_df['Output0'])
    test_df['Target0'] = pd.to_numeric(test_df['Target0'])

    # Calculate R², RMSE, and MAE for training data
    y_true_train = train_df['Target0']
    y_pred_train = train_df['Output0']
    r2_train = r2_score(y_true_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_true_train, y_pred_train))
    mae_train = mean_absolute_error(y_true_train, y_pred_train)

    # Calculate R², RMSE, and MAE for testing data
    y_true_test = test_df['Target0']
    y_pred_test = test_df['Output0']
    r2_test = r2_score(y_true_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
    mae_test = mean_absolute_error(y_true_test, y_pred_test)

    # Combine the metrics into a title string
    title = (
        f"Training Data: R² = {r2_train:.2f}, RMSE = {rmse_train:.2f}, MAE = {mae_train:.2f}\n"
        f"Testing Data: R² = {r2_test:.2f}, RMSE = {rmse_test:.2f}, MAE = {mae_test:.2f}"
    )

    # Create the plot
    sns.set(style="white")  # Set white background and no grid
    plt.figure(figsize=(8, 6))

    # Plot the training data regression plot
    sns.regplot(x='Output0', y='Target0', data=train_df, scatter_kws={'s': 50, 'color': 'blue'}, line_kws={'color': 'red'}, label="Training Data")

    # Plot the testing data regression plot
    sns.regplot(x='Output0', y='Target0', data=test_df, scatter_kws={'s': 50, 'color': 'green'}, line_kws={'color': 'orange'}, label="Testing Data")

    # Set the title with error metrics
    plt.title(title, fontsize=14)

    # Labels
    plt.xlabel('Predicted UCS (MPa)')
    plt.ylabel('Actual UCS (MPa)')

    # Display legend
    plt.legend()

    # Show the plot
    st.pyplot(plt)

# Streamlit app
st.title("Regression Plot with Error Metrics")

# File uploader for training and testing files
train_file = st.file_uploader("Upload the training CSV file", type="csv")
test_file = st.file_uploader("Upload the testing CSV file", type="csv")

if train_file and test_file:
    st.write("Files are uploaded, generating plot...")
    plot_regression(train_file, test_file)
else:
    st.write("Please upload both training and testing CSV files.")
