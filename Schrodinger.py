import torch
import torch.nn as nn
from scipy.special import hermite

import numpy as np
import matplotlib.pyplot as plt


## For further inspiration, take a look at
## https://github.com/nanditadoloi/PINN/blob/main/solve_PDE_NN.ipynb


###########
## MODEL ##
###########
class PINN(nn.Module):
    def __init__(self, energy):
        super(PINN, self).__init__()
        self.hidden_layer1 = nn.Linear(1,40)
        self.hidden_layer2 = nn.Linear(40,20)
        self.hidden_layer3 = nn.Linear(20, 10)
        self.hidden_layer4 = nn.Linear(10, 5)
        self.output_layer = nn.Linear(5,1)

        ## Diff. Eq. Params ##
        self.omega = torch.rand(1, requires_grad=True)
        self.E = energy

        params = list(self.parameters())
        params.append(self.omega)

        ## TODO: define the optimizer
        self.optimizer = torch.optim.Adam(params, lr=0.001)
        ## TODO: define the loss function
        self.criterion = nn.MSELoss()
    def forward(self, x):
        inputs = torch.cat([x], axis=1)
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        ## For regression, no activation is used in output layer ##
        output = self.output_layer(layer4_out)
        return output

    def diff_eq(self, x):
        """
        This method returns an implicit function which set = 0 gives the differential equation
        :param t: the times used to estimate the value of the function f(t)
        :return: the implicit function f(t)
        """
        ## TODO: predict the value of psi
        psi = self(x)
        ## TODO: derive first and second order of psi
        psi_x = torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        psi_xx = torch.autograd.grad(psi_x.sum(),x, create_graph=True)[0]
        ## TODO: return the implicit function for the differential equation
        return -0.5*psi_xx + self.omega**2*x**2/2*psi - self.E*psi

    def train_step(self, diff_eq_x, dataset_x, amplitudes):
        """
        This method train the NN model using the data loss and the differential equation loss
        :param diff_eq_x: tensor of positions used to estimate the differential equation
        :param dataset_x: tensor of positions used to estimate the velocities using the NN
        :param amplitudes: tensor of "labels" used to estimate the data loss
        :return: the total loss: loss_total = loss_data + loss_diff_eq
        """
        self.optimizer.zero_grad()

        # TODO: estimate the data loss (take inspiration from the parabola.py or the Ising files)
        loss_data = self.criterion(self(dataset_x), amplitudes.reshape(-1,1))
        # Here we estimate the differential equation loss
        f_out = self.diff_eq(diff_eq_x) # output of f(x,t)
        zeros_diff_eq = torch.zeros(size=(500, 1), requires_grad=False, dtype=torch.float32)
        loss_diff_eq = self.criterion(f_out, zeros_diff_eq)

        # TODO: sum the two losses (loss_total)
        loss_total = loss_data+loss_diff_eq
        ## TODO: compute the total backpropagation
        loss_total.backward()
        self.optimizer.step()
        return loss_total.item()


def wavefunc(x, n, omega):
    H = hermite(n)
    return (2 / np.pi)**(1/4) * np.exp(-omega / 2 * x ** 2) / np.sqrt(2**n * np.math.factorial(n)) * H(x)



##############
## TRAINING ##
##############
def main():
    ## Parameters ##
    energy = 2.75
    x_min, x_max = -6, 6

    dataset_ = np.load("SCHRODINGER_dataset.npz")
    x_data, psi_dataset = dataset_["x_data"], dataset_["psi_data"]

    x_data = torch.tensor(x_data, requires_grad=True, dtype=torch.float32)
    psi_dataset = torch.tensor(psi_dataset, requires_grad=True, dtype=torch.float32)

    ## TODO: Instantiate the model
    net = PINN(energy)

    ## TODO: Train the model, define number of epochs
    n_epochs = 10000
    batch_size = 500
    for epoch in range(n_epochs):

        ## Differential Equation data ##
        x_eq = (x_min - x_max) * torch.rand(size=(batch_size, 1), requires_grad=True, dtype=torch.float32) + x_max

        loss = net.train_step(diff_eq_x=x_eq, dataset_x=x_data, amplitudes=psi_dataset)

        print(epoch+1, "Training Loss:", loss)


    ############## TEST ##############
    ## Generate a sequence of times ##
    ## Plot the model predictions   ##
    ## Plot the true dynamic curve  ##
    ## Plot the dataset of (t,v)    ##
    ##################################
    with torch.autograd.no_grad():

        #print("Learned omega: {}".format(net.omega))
        n_pred = round(energy/net.omega.item() - .5)
        omega = net.omega.item()
        print("Learned omega: {}".format(omega))
        print("Energy level: {}".format(n_pred))

        ## Data to plot ##
        x_test = np.arange(x_min, x_max, 1/200).reshape((x_max-x_min)*200, 1)
        psi_true = np.array([wavefunc(x, n_pred, omega) for x in x_test])
        ## TODO: predict the velocities from t_test in order to plot them
        psi_test = net(torch.tensor(x_test, dtype = torch.float32))


    ## Plotting ##
    fig, ax = plt.subplots()
    ax.plot(x_test, psi_test.detach().numpy(), label="Model")
    ax.plot(x_test, psi_true, '--', label="Ground Truth")
    ax.plot(x_data.detach().numpy(), psi_dataset.detach().numpy(), 'o', label="Dataset")

    ax.legend(loc='upper right', shadow=True)
    ax.set_xlabel('Space coords (a.u.)', weight='bold')
    ax.set_ylabel('Probability amplitude', weight='bold')
    ax.set_title('Wavefunction', fontsize=22, weight='bold')
    plt.show()


if __name__=="__main__":
    main()
