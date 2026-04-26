import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt


## For further inspiration, take a look at
## https://github.com/nanditadoloi/PINN/blob/main/solve_PDE_NN.ipynb


###########
## MODEL ##
###########
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden_layer1 = nn.Linear(1,5)
        self.output_layer = nn.Linear(5,1)

        ## TODO: instantiate the parameters randomly in a proper range
        ## Use a torch.rand generator!
        self.tau = torch.rand(1, requires_grad=True)
        self.V = torch.rand(1, requires_grad=True)

        ## TODO: add the tau and V constants to the params of the network
        params = list(self.parameters())
        params.append(self.tau)
        params.append(self.V)

        ## TODO: define the optimizer
        self.optimizer = torch.optim.Adam(params, lr = 0.001)

        ## TODO: define the loss function
        self.criterion = nn.MSELoss()

    def forward(self, t):
        inputs = torch.cat([t], axis=1)
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        ## For regression, no activation is used in output layer ##
        output = self.output_layer(layer1_out)
        return output

    def diff_eq(self, t):
        """
        This method returns an implicit function which set = 0 gives the differential equation
        :param t: the times used to estimate the value of the function F(t)
        :return: the implicit function F(t)
        """
        ## TODO: predict the velocities
        v = self(t)
        ## TODO: derive the velocities with respect to t (first order derivative dv/dt)
        dv_dt = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        ## TODO: return the implicit function for the differential equation
        return dv_dt + (v - self.V)/self.tau

    def train_step(self, diff_eq_times, dataset_times, dataset_vel):
        """
        This method train the NN model using the data loss and the differential equation loss
        :param diff_eq_times: tensor of times used to estimate the differential equation
        :param dataset_times: tensor of times used to estimate the velocities using the NN
        :param dataset_vel: tensor of "labels" used to estimate the data loss
        :return: the total loss: loss_total = loss_data + loss_diff_eq
        """
        self.optimizer.zero_grad()

        ## TODO: estimate the data loss (take inspiration from the parabola.py or the Ising files)
        loss_data = self.criterion(self(dataset_times), dataset_vel.unsqueeze(1))
        # Here we estimate the differential equation loss
        f_out = self.diff_eq(diff_eq_times)  # output of f(x,t)
        zeros_diff_eq = torch.zeros(size=(500, 1), requires_grad=False, dtype=torch.float32)
        loss_diff_eq = self.criterion(f_out, zeros_diff_eq)

        ## TODO: sum the two losses (loss_total)
        t0 = torch.zeros(1, 1, dtype=torch.float32)
        loss_ic = self.criterion(self(t0), torch.tensor([[6.0]]))
        loss_total = loss_data + loss_diff_eq + loss_ic

        #loss_total = loss_data + loss_diff_eq
        ## TODO: compute the totale backpropagation
        loss_total.backward()
        self.optimizer.step()
        return loss_total.item()
##############
## TRAINING ##
##############
def main():
    ## Data upload ##
    dataset_ = np.load("STOKES_dataset.npz")
    t_dataset, v_dataset = dataset_["times"], dataset_["velocities"]

    ## Parameters ##
    times_min, times_max = 0.0, 4.

    ## Dataset in tensor format ##
    times_data = torch.tensor(t_dataset, requires_grad=True, dtype=torch.float32)
    velocities = torch.tensor(v_dataset, requires_grad=True, dtype=torch.float32)

    ## TODO: Instantiate the model
    net = PINN()

    ## TODO: Train the model, define number of epochs and batch size
    n_epochs = 10000
    batch_size = 500
    for epoch in range(n_epochs):

        ## Differential Equation data ##
        times_eq = (times_min - times_max) * torch.rand(size=(batch_size, 1), requires_grad=True, dtype=torch.float32) + times_max

        loss = net.train_step(diff_eq_times=times_eq, dataset_times=times_data, dataset_vel=velocities)

        print(epoch+1, "Training Loss:", loss)


    ############## TEST ##############
    ## Generate a sequence of times ##
    ## Plot the model predictions   ##
    ## Plot the true dynamic curve  ##
    ## Plot the dataset of (t,v)    ##
    ##################################
    with torch.autograd.no_grad():

        print("Learned tau: {}".format(net.tau))
        print("Learned V: {}".format(net.V))

        ## Data to plot ##
        t_test = np.arange(0, times_max, 1/200).reshape(800, 1)
        v_true = np.array([2 * np.exp(-t / net.tau) + 4 for t in t_test])
        ## TODO: predict the velocities from t_test in order to plot them
        ## use .detach().numpy() method to convert torch.tensor to numpy
        v_test = net(torch.tensor(t_test, dtype = torch.float32))


    ## Plotting ##
    fig, ax = plt.subplots()
    ax.plot(t_test, v_test.detach().numpy(), label="Model")
    ax.plot(t_test, v_true, '--', label="Ground Truth")
    ax.plot(t_dataset, v_dataset, 'o', label="Dataset")

    ax.legend(loc='upper right', shadow=True)
    ax.set_xlabel('Time (s)', weight='bold')
    ax.set_ylabel('Velocity (m/s)', weight='bold')
    ax.set_title('Stokes Dynamics', fontsize=22, weight='bold')
    plt.show()


if __name__=="__main__":
    main()
