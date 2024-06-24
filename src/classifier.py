from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def Classifier(real_train_labels, real_train_images, generated_labels, generated_samples):

    # ------------
    # Form training set from real images
    # ------------

    k = 5
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit (real_train_images, real_train_labels)
    predicted_labels = knn.predict(generated_samples)

    accuracy = np.mean(predicted_labels == generated_labels)

    print(f'Classification accuracy: {accuracy}')