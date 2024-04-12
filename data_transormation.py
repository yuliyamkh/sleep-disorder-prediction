import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def save_transformed_data(dataset, filename):
    """
    Saves data into a .csv-file.
    :param dataset: pandas DataFrame
    :param filename: str
    :return: None
    """

    # Create a folder if it does not exist
    folder = 'data'
    if not os.path.exists(folder):
        os.makedirs('data')

    # Concatenate the folder path and file name
    file_path = os.path.join(folder, filename)
    dataset.to_csv(file_path, index=False)


def transform_feature(feature_to_transform, feature_type, encoder=None):
    """
    Transforms a categorical feature into a numerical representation.
    :param feature_to_transform: a feature array
    :param feature_type: str ('ordinal' or 'nominal')
    :param encoder: sklearn preprocessing encoder
    :param categories: an array of categories for the encoder
    :return:
    """

    if feature_type == 'ordinal':
        feature_to_transform = encoder.fit_transform(feature_to_transform)
    elif feature_type == 'nominal':
        feature_to_transform = pd.get_dummies(feature_to_transform, dtype=int)

    return feature_to_transform


if __name__ == '__main__':
    data = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")
    data['Sleep Disorder'] = data['Sleep Disorder'].fillna('No Disorder')
    data['BMI Category'] = data['BMI Category'].replace('Normal Weight', 'Normal')
    print("Info: ", data.info(), '\n')

    # EXPLORE DATA
    blood_pressure_vals = data['Blood Pressure'].value_counts()
    bmi_category_vals = data['BMI Category'].value_counts()
    quality_of_sleep_vals = data['Quality of Sleep'].value_counts()
    y_vals = data['Sleep Disorder'].value_counts()

    print(blood_pressure_vals, '\n')
    print(bmi_category_vals, '\n')
    print(quality_of_sleep_vals, '\n')
    print(y_vals, '\n')

    # NORMALIZE DATA
    # Transform the BMI category (ordinal) feature into ordinal categories
    ordinal_encoder = OrdinalEncoder(categories=[['Normal', 'Overweight', 'Obese']])
    bmi_category_reshaped = data['BMI Category'].values.reshape(-1, 1)
    data['BMI Category Rating'] = transform_feature(bmi_category_reshaped, 'ordinal', ordinal_encoder)

    print(f"Shapes:\ninitial: {len(data['BMI Category'].values)}, after reshaping: {bmi_category_reshaped.shape}")

    # Transform the Gender (nominal) feature into one-hot vectors
    ohe_gender = transform_feature(data['Gender'], 'nominal')
    data = data.join(ohe_gender)

    # Transform the Occupation (nominal) feature into one-hot vectors
    ohe_occupation = transform_feature(data['Occupation'], 'nominal')
    data = data.join(ohe_occupation)

    # Extract only numerical features
    data = data[['Person ID', 'Age', 'Sleep Duration', 'Quality of Sleep',
                 'Physical Activity Level', 'Stress Level', 'Heart Rate',
                 'Daily Steps', 'BMI Category Rating', 'Female', 'Male',
                 'Accountant', 'Doctor', 'Engineer', 'Lawyer', 'Manager',
                 'Nurse', 'Sales Representative', 'Salesperson', 'Scientist',
                 'Software Engineer', 'Teacher', 'Sleep Disorder']].copy()

    # Get info after transformations
    data.info()

    # SAVE TRANSFORMED DATA
    save_transformed_data(data, 'transformed_Sleep_health_and_lifestyle_dataset')

