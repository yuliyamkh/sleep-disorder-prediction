import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
    data = pd.read_csv('data/data_balanced_classes')
    features, targets = data.iloc[:, :-1], data.iloc[:, -1]

    # Scale features before model fit
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Transform targets into a numerical form
    # targets = pd.get_dummies(targets, dtype=int)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.1, random_state=42)

    # Define a model
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

    # Define the model evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # Evaluate the model and collect the scores
    n_scores = cross_val_score(model, X_train, y_train,
                               scoring='accuracy',
                               cv=cv,
                               n_jobs=-1)

    # Report the model performance
    print(f"Mean Accuracy: {np.round(np.mean(n_scores), 2)}, {np.round(np.std(n_scores), 2)}")
