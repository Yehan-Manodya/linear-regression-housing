from sklearn.linear_model import LinearRegression
from preprocess import load_and_preprocess

def train_model():
    # Load processed data
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # Create Linear Regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Return trained model and test data
    return model, X_test, y_test
