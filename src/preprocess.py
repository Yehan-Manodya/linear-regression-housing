import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess():
    # 1. Load dataset
    df = pd.read_csv("data/housing.csv")

    # 2. Handle missing values (fill NaN with column mean)
    df = df.fillna(df.mean(numeric_only=True))

    # 3. Convert categorical column to numerical
    df = pd.get_dummies(df, columns=["ocean_proximity"])

    # 4. Separate features and target
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
