import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.core.fromnumeric import argmax
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, auc, balanced_accuracy_score


def anomaly_characterization(params, testing_data):
    """
    Such function builds up a comparison graph between average behaviours of anomalies and normal signals.

    :param params:
    :param testing_data:
    :return:
    """
    n_anomalies = params[0]
    n_samples = params[1]
    n_components = params[2]
    ## Create the pandas dataframe from the features
    df = pd.DataFrame(testing_data)
    ## Let's see how an anomaly appears:
    plt.figure(figsize=(8, 5))
    plt.grid(True)
    plt.title("Comparison between average behaviours of anomalies and normal signals")
    plt.plot(df.iloc[:n_samples - n_anomalies, :n_components].mean(numeric_only=True), lw=2, label='Normal')
    plt.plot(df.iloc[-1*n_anomalies:-1, :n_components].mean(numeric_only=True), lw=2, ls='--', label='Anomaly')
    plt.plot(df.iloc[:, :n_components].mean(numeric_only=True), lw=2, ls=':', label='All')
    plt.xlabel('Components')
    plt.ylabel('Signal values')
    plt.legend()
    plt.show()

    ## Error's order of magnitude:
    normal_mean = df.iloc[:n_samples - n_anomalies, :n_components].mean(numeric_only=True).to_numpy()
    anomaly_mean = df.iloc[-1*n_anomalies:-1, :n_components].mean(numeric_only=True).to_numpy()
    normal = testing_data[4]

    error_among_normal_and_anomaly = np.mean(np.abs(anomaly_mean - normal_mean), axis=0)
    error_among_normal_and_normal = np.mean(np.abs(normal - normal_mean), axis=0)
    print("The error between the normal mean and the anomaly mean is:", error_among_normal_and_anomaly)
    print("An example of error between the normal mean and a single normal signal is:", error_among_normal_and_normal)



def first_three_components_plot(y, labels):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([row[0] for row in y], [row[1] for row in y], [row[2] for row in y], c=labels)
    #ax.scatter([row[0] for row in y], [row[1] for row in y], [0 for row in y], c=labels)

    ax.set_xlabel('1st component')
    ax.set_ylabel('2nd component')
    ax.set_zlabel('3rd component')
    plt.show()



def cumulative_variance_graph(n_components, cumulative_variance):
    ## Plot the explained variance
    plt.figure(figsize=(8, 5))
    plt.grid(True)
    plt.plot(range(1, n_components + 1), cumulative_variance, marker='o', linestyle='-', color='b', label='Cumulative variance')
    plt.title("PCA cumulative explained variance")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative variance")
    plt.ylim(0,1)
    plt.show()



def reconstruction_error_distribution(reconstruction_error, labels):
    plt.figure(figsize=(8, 5))
    plt.title("Reconstruction error distribution")
    plt.grid(True)
    plt.hist(reconstruction_error[labels == 0], bins=50, alpha=0.5, label='Normal')
    plt.hist(reconstruction_error[labels == 1], bins=50, alpha=0.9, label='Anomaly')
    plt.xlabel('Reconstruction error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()



def plot_ROC(RE, labels):
    """
    This function plots the ROC curve
    :param RE: the reconstruction errors
    :param labels: the true labels
    :return:
    """
    ## gammas contains different thresholds used to discriminate anomalies from normal data
    gammas = np.linspace(np.min(RE), np.max(RE), 100)

    X, Y = [], []

    for gamma in gammas:
        '''
        Evaluate TPR and FPR for every threshold gamma
        '''
        anomalies = RE > gamma

        TN, FP, FN, TP = confusion_matrix(labels, anomalies, labels=[0, 1]).ravel()

        sensitivity = TP / (TP + FN)
        specificity = FP / (TN + FP)

        Y.append(sensitivity)
        X.append(specificity)

    ## Add 0 threshold (used to plot a nicer curve only)
    X.append(0)
    Y.append(0)

    ## Plot the ROC curve and save the image
    plt.rcParams['figure.figsize'] = [8, 5]
    plt.figure(figsize=(8, 5))
    plt.grid(True)
    plt.plot(X, Y, linestyle='--')
    plt.title('ROC curve')
    plt.scatter(X, Y)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.savefig("ROC.png")

    return X, Y



def balance_accuracy_plot(reconstruction_error, labels):
    percentile = np.arange(80, 99.5, 0.1)
    balanced_accuracy = []
    for p in percentile:
        threshold = np.percentile(reconstruction_error, p)  # Adjust percentile as needed
        anomalies_indices = np.where(reconstruction_error > threshold)[0]

        normal = 0
        fraud = 0
        for i in anomalies_indices:
            if labels[i]:
                normal = normal + 1
            else:
                fraud = fraud + 1

        anomalies = reconstruction_error > threshold
        TN, FP, FN, TP = confusion_matrix(labels, anomalies, labels=[0, 1]).ravel()

        precision = TP / (FP + TP)
        recall = TP / (TP + FN)

        balanced_accuracy.append(0.5*(precision + recall))

    p_max = argmax(balanced_accuracy)
    print('The maximum value of th balanced accuracy is ', max(balanced_accuracy), ' and the corresponding percentile is at ', percentile[p_max] )
    plt.figure(figsize=(8,5))
    plt.grid(True)
    plt.plot(percentile, balanced_accuracy, lw=2, c='r')
    plt.xlabel('Percentile')
    plt.ylabel('Balanced accuracy')
    plt.show()



def main():

    ## The following dictionary contains: the number of anomalies in the test dataset (the training dataset cannot and should not contain any),
    ## the number of samples, the number of components for each element in the test dataset, and the corresponding file.

    dataset_dict = {'Ising': [129, 2649, 25,'SpinsAD.npz'],
                    'HeartBeats': [114, 2891, 170,'HeartBeatsAD.npz']}

    ## TODO: Repeat the same exercise for the heartbeats dataset.
    system_name = 'HeartBeats'

    ## Load ECG dataset (contains values of the signals)
    #raw_data = np.load(dataset_dict[system_name][-1])
    raw_data = np.load('/home/timofey/AI for physics/23-04-2026 - Anomaly detection/HeartBeatsAD.npz')
    
    training_data, testing_data, labels = raw_data["training_data"], raw_data["testing_data"], raw_data["labels"]

    anomaly_characterization(params=dataset_dict[system_name], testing_data=testing_data)

    ## TODO: Define a PCA model using a suitable number of components (max_n_components > n_components > 0)
    n_components = 10
    pca = PCA(n_components)

    ## TODO: fit the model using training data
    pca.fit(training_data)

    ## TODO: apply PCA on testing data reduced_input = PCA(data)
    y = pca.transform(testing_data)

    ## Let's have a look at the graphical representation of the first three components of the reduced input.
    ## Is such dimensionality reduction already enough to distinguish anomalies?
    if n_components >= 3:
        first_three_components_plot(y, labels)

    ## Calculate explained variance and cumulative explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    ## Let's see how the cumulative variance evolves increasing the number of principal components:
    cumulative_variance_graph(n_components, cumulative_variance)

    ## TODO: apply the inverse transformation to reconstruct the data reconstructed_input = inverse_PCA(reduced_input)
    reconstructed_input = pca.inverse_transform(y)

    ## TODO: evaluate the Reconstruction Error (Mean Absolute Error)
    reconstruction_error = np.mean(np.abs(reconstructed_input-testing_data), axis = 1)

    ## Let's plot the reconstruction error distribution:
    reconstruction_error_distribution(reconstruction_error, labels)

    ## Let's plot the ROC curve:
    X, Y = plot_ROC(reconstruction_error, labels)
    print("The AUC is {:.4f}".format(auc(X, Y)))

    ## Plot the balance accuracy and the threshold at which its maximum value corresponds.
    balance_accuracy_plot(reconstruction_error, labels)



if __name__ == "__main__":
    main()
