import numpy as np
import torch
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, auc


def plot_ROC(RE, labels):
    """
    This function plots the ROC curve
    :param RE: the reconstruction errors
    :param labels: the true labels
    :return:
    """
    # gammas contains different thresholds used to discriminate anomalies from normal data
    gammas = np.linspace(torch.min(RE.detach()), torch.max(RE.detach()), 500)

    X, Y = [], []

    for gamma in gammas:
        '''
        Evaluate TPR and FPR for every threshold gamma
        '''
        anomalies = RE > gamma

        TN, FP, FN, TP = confusion_matrix(labels, anomalies).ravel()

        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)

        Y.append(TPR)
        X.append(FPR)

    # Add 0 threshold (used to plot a nicer curve only)
    X.append(0)
    Y.append(0)

    plt.rcParams['figure.figsize'] = [10, 10]

    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.plot(X, Y, linestyle='--')
    plt.scatter(X, Y)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.savefig("ROC.png")
    plt.show()

    return X, Y

class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # TODO: define the model (autoencoder)

        # This is the encoder (compress the information)
        self.encoder = nn.Sequential(
            nn.Linear(input_size,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU()
        )
        # This is the decoder (reconstruct the compressed information)
        self.decoder = nn.Sequential(
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

        # TODO: define the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)
        # TODO: define the loss (MSE)
        self.criterion = nn.MSELoss()
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def train_step(self, inputs):
        # TODO: define the training step
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # forward + backward + optimize
        outputs = self.forward(inputs)
        loss = self.criterion(inputs, outputs)
        loss.backward()
        self.optimizer.step()

        return loss.item()

def main():

    # Load ECG dataset (contains values of the signals)
    raw_data = np.load('/home/timofey/AI for physics/23-04-2026 - Anomaly detection/HeartBeatsAD.npz')

    training_data, testing_data, labels = raw_data["training_data"], raw_data["testing_data"], raw_data["labels"]

    # Convert to Tensors
    training_data = torch.tensor(training_data, dtype=torch.float32)
    test_inputs = torch.tensor(testing_data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    # Get the number of input feature
    n_features = training_data.shape[1]

    # TODO: Create a pytorch dataset
    dataset_train = TensorDataset(training_data)
    loader_train = DataLoader(dataset_train, batch_size=64, shuffle= True)

    # TODO: Define the autoencoder
    autoencoder_model = Autoencoder(n_features)

    # Train the autoencoder on normal data only
    n_epochs = 4
    for epoch in range(n_epochs):
        for input_data in loader_train:
            # TODO: train the model
            loss = autoencoder_model.train_step(input_data[0])

    with torch.no_grad():
        reconstructed_input = autoencoder_model(test_inputs)

    # TODO: evaluate the reconstruction error "recostructionError" (use torch)
    recostructionError = torch.mean(torch.abs(test_inputs - reconstructed_input), dim = 1)

    X, Y = plot_ROC(recostructionError, labels)

    print("The AUC is {:.4f}".format(auc(X, Y)))

if __name__ == "__main__": main()