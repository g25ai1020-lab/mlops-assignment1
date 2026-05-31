import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def load_data():
    """
    Load Boston Housing dataset manually from CMU server
    
    Returns:
    - df: pandas DataFrame with features and target
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    
    # Split into data and target
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    
    # Feature names based on original dataset
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target  # MEDV is the target variable (house prices)
    
    return df

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Split and preprocess data - standardize features
    
    Parameters:
    - df: pandas DataFrame with features and target
    - test_size: proportion of test set (float between 0 and 1)
    - random_state: seed for reproducibility
    
    Returns:
    - X_train_scaled: standardized training features
    - X_test_scaled: standardized test features
    - y_train: training target values
    - y_test: test target values
    """
    # Separate features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardize features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_model(model, X_train, y_train):
    """
    Train a machine learning model
    
    Parameters:
    - model: sklearn model object (e.g., DecisionTreeRegressor, KernelRidge)
    - X_train: training features
    - y_train: training target values
    
    Returns:
    - model: trained model object
    """
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance using Mean Squared Error (MSE)
    
    Parameters:
    - model: trained sklearn model
    - X_test: test features
    - y_test: test target values
    
    Returns:
    - mse: Mean Squared Error on test set
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def print_results(model_name, mse):
    """
    Print model results in a formatted way
    
    Parameters:
    - model_name: name of the model
    - mse: Mean Squared Error value
    """
    rmse = mse ** 0.5
    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)
    print(f"Mean Squared Error (MSE):      {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("=" * 60 + "\n")
