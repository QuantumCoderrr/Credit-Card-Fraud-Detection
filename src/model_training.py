from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(df_sampled):
    X = df_sampled.drop("Class", axis=1)
    y = df_sampled["Class"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("\nTraining and testing split done.")

    # Train the Random Forest model
    print("\nTraining the Random Forest model...")
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed.")

    return model, X_test, y_test
