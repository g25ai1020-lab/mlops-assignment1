from sklearn.tree import DecisionTreeRegressor
from misc import load_data, preprocess_data, train_model, evaluate_model, print_results

def main():
    print("\n" + "=" * 60)
    print("DECISION TREE REGRESSOR - BOSTON HOUSING PREDICTION")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data...")
    df = load_data()
    print(f"      Dataset shape: {df.shape}")
    print(f"      Features: {list(df.columns[:-1])}")
    
    # Preprocess data
    print("\n[2/4] Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print(f"      Training set size: {X_train.shape}")
    print(f"      Test set size: {X_test.shape}")
    
    # Create and train model
    print("\n[3/4] Training Decision Tree Regressor...")
    model = DecisionTreeRegressor(random_state=42, max_depth=10)
    model = train_model(model, X_train, y_train)
    print("      Model trained successfully!")
    
    # Evaluate model
    print("\n[4/4] Evaluating model on test set...")
    mse = evaluate_model(model, X_test, y_test)
    
    print_results("Decision Tree Regressor", mse)

if __name__ == "__main__":
    main()
