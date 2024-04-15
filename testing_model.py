import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from data_exploration import save_plot_as_image


if __name__ == '__main__':
    # Load the saved model
    model = joblib.load('model/multinomial_logistic_regression_model.joblib')

    # Load data
    features = pd.read_csv('data/scaled_features.csv').values
    targets = pd.read_csv('data/target_classes.csv')

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42, shuffle=True)

    # Check the performance of the model on train data
    y_pred_train = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print('\n')
    print(f'Accuracy on train data: {np.round(accuracy_train, 2)}')

    # Check the performance of the model on test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('\n')
    print(f'Accuracy on test data: {np.round(accuracy, 2)}')

    # Generate classification report
    print('\n')
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Compute confusion matrix of test target classes and predictions produced by the model
    cm = confusion_matrix(y_test, y_pred)
    classes = np.unique(y_test)

    # Plot confusion matrix
    # plt.figure(figsize=(8, 6))
    sns.set(font_scale=1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    save_plot_as_image('images', 'confusion_matrix')
    plt.show()
