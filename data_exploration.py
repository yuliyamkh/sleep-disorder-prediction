import pandas as pd
import os
import matplotlib.pyplot as plt
from data_transormation import save_transformed_data


def save_plot_as_image(folder, image_name):
    """
    Save an image into the specified filepath.
    :param folder: name of the folder, str
    :param image_name: name of the image file, str
    :return: None
    """

    if not os.path.exists(folder):
        os.makedirs(folder)

    filepath = os.path.join(folder, image_name)
    plt.savefig(filepath)


if __name__ == '__main__':
    data = pd.read_csv('data/transformed_Sleep_health_and_lifestyle_dataset')
    features = data.iloc[:, :-1]
    targets = data.iloc[:, -1]

    # Distribution of the target variable sleep disorder
    category_colors = {'No Disorder': 'gray', 'Sleep Apnea': 'green', 'Insomnia': 'orange'}
    targets_counts = targets.value_counts()

    targets.value_counts().plot(kind='bar',
                                color=[category_colors[disorder] for disorder in targets_counts.index],
                                title='Distribution of Sleep Disorder target classes',
                                xlabel='', ylabel='Counts'
                                )

    plt.xticks(rotation=45, fontsize=7)
    plt.tight_layout()
    save_plot_as_image(folder='images', image_name='classes_unbalanced')
    plt.show()

    # The plot shows that target classes are imbalanced
    # The majority class is class 1: No Disorder
    # To balance target classes, I remove some observations
    # of the majority class using an undersampling technique
    no_disorder_undersampling = data[data['Sleep Disorder'] == 'No Disorder'].sample(targets_counts['Sleep Apnea'])
    data_after_undersampling = pd.concat([no_disorder_undersampling,
                                          data[data['Sleep Disorder'] == 'Sleep Apnea'],
                                          data[data['Sleep Disorder'] == 'Insomnia']])

    data_after_undersampling['Sleep Disorder'].value_counts().plot(kind='bar',
                                                                   color=[category_colors[disorder] for disorder in targets_counts.index],
                                                                   title='Distribution of Sleep Disorder target classes',
                                                                   xlabel='', ylabel='Counts'
                                                                   )

    plt.xticks(rotation=45, fontsize=7)
    plt.tight_layout()
    save_plot_as_image(folder='images', image_name='classes_balanced')
    plt.show()

    save_transformed_data(data_after_undersampling, 'data_balanced_classes')

