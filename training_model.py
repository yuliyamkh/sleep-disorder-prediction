import matplotlib.pyplot as plt
import seaborn as sns
from data_exploration import save_plot_as_image
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def tune_penalty_hyperparameter(parameter_array):
    """
    Tune the regularization parameter (L2).
    """
    models = {}
    for p in parameter_array:
        if p == 0.0:
            models[p] = LogisticRegression(multi_class='multinomial',
                                           solver='lbfgs',
                                           penalty=None,
                                           max_iter=1000)
        else:
            models[p] = LogisticRegression(multi_class='multinomial',
                                           solver='lbfgs',
                                           penalty='l2',
                                           C=p,
                                           max_iter=1000)

    return models


def evaluate_model(mdl, X, y):
    """
    Evaluate a model using cross-validation.
    """
    cross_val = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(mdl, X, y, scoring='accuracy', cv=cross_val, n_jobs=-1)

    return scores


if __name__ == '__main__':
    data = pd.read_csv('data/data_balanced_classes')
    features, targets = data.iloc[:, :-1], data.iloc[:, -1]

    # Scale features before model fit
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Transform targets into a numerical form
    # targets = pd.get_dummies(targets, dtype=int)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42, shuffle=True)

    # Define a model
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

    # Define the model evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10,
                                 n_repeats=3,
                                 random_state=1)

    # Evaluate the model and collect the scores
    n_scores = cross_val_score(model, X_train, y_train,
                               scoring='accuracy',
                               cv=cv,
                               n_jobs=-1)

    # Report the model performance
    print(f"Mean Accuracy: {np.round(np.mean(n_scores), 2)}, {np.round(np.std(n_scores), 2)}")
    print('\n')

    # Tune penalty for multinomial logistic regression
    models = tune_penalty_hyperparameter(parameter_array=[0.0, 0.0001, 0.001, 0.01, 0.1, 1.0])
    results, names = [], []
    for name, model in models.items():
        scores = evaluate_model(model, X_train, y_train)
        results.append(scores)
        names.append(name)

        print(f"MLG model with C = {name}: {np.round(np.mean(scores), 2)}, {np.round(np.std(scores), 2)}")

    plt.boxplot(results, labels=names, showmeans=True)
    save_plot_as_image(folder='images', image_name='L2_regularization')
    plt.show()

    # Train model with the best penalty parameter
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=0.001)

    # Fit the model
    model.fit(X_train, y_train)

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

