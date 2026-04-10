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
    def __init__(self, tau, V):
        super(PINN, self).__init__()
        self.hidden_layer1 = nn.Linear(1,5)
        self.hidden_layer2 = nn.Linear(5,5)
        self.hidden_layer3 = nn.Linear(5,5)
        self.hidden_layer4 = nn.Linear(5,5)
        self.hidden_layer5 = nn.Linear(5,5)
        self.output_layer = nn.Linear(5,1)

        ## Diff. Eq. Params ##
        self.tau, self.V = tau, V

        ## TODO: define the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.01)

        ## TODO: define the loss function

        self.criterion = nn.MSELoss()
    def forward(self, t):
        inputs = torch.cat([t], axis=1)
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
        ## For regression, no activation is used in output layer ##
        output = self.output_layer(layer5_out)
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
        dv_dt = torch.autograd.grad(v,t,grad_outputs=torch.ones_like(v), create_graph = True)[0]
        ## TODO: return the implicit function for the differential equation
        return (dv_dt + (v - self.V)/self.tau)
    def train_step(self, diff_eq_times, dataset_times, dataset_vel):
        """
        This method train the NN model using the data loss and the differential equation loss
        :param diff_eq_times: tensor of times used to estimate the differential equation
        :param dataset_times: tensor of times used to estimate the velocities using the NN
        :param dataset_vel: tensor of "labels" used to estimate the data loss
        :return: the total loss: loss_total = loss_data + loss_diff_eq
        """
        self.optimizer.zero_grad()

        # TODO: estimate the data loss (take inspiration from the parabola.py or the Ising files)
        outputs = self(dataset_times)
        labels = dataset_vel.unsqueeze(1)
        loss_data = self.criterion(outputs, labels)
        # Here we estimate the differential equation loss
        f_out = self.diff_eq(diff_eq_times)  # output of f(x,t)
        zeros_diff_eq = torch.zeros(size=(f_out.shape[0], 1), requires_grad=False, dtype=torch.float32)
        loss_diff_eq = self.criterion(f_out, zeros_diff_eq)

        # TODO: sum the two losses (loss_total)
        loss_total = loss_data + loss_diff_eq
        # TODO: compute the total backpropagation

        loss_total.backward()
        self.optimizer.step()
        return loss_total.item()
##############
## TRAINING ##
##############
def main():
    ## Data upload ##
    dataset_ = np.load("./STOKES_dataset.npz")
    t_dataset, v_dataset = dataset_["times"], dataset_["velocities"]

    ## Parameters ##
    tau_ = 2.0
    V_ = 2.0
    times_min, times_max = 0.0, 2 * tau_

    ## Dataset in tensor format ##
    times_data = torch.tensor(t_dataset, requires_grad=True, dtype=torch.float32)
    velocities = torch.tensor(v_dataset, requires_grad=True, dtype=torch.float32)

    ## TODO: Instantiate the model
    net = PINN(tau=tau_,V=V_)

    ## TODO: Train the model and define number of epochs and batch size
    n_epochs = 2000
    batch_size = 256
    for epoch in range(n_epochs):
        ## Differential Equation data ##
        times_eq = (times_min - times_max) * torch.rand(size=(batch_size, 1),
                                                        dtype=torch.float32) + times_max
        times_eq.requires_grad_(True)
        loss = net.train_step(diff_eq_times=times_eq, dataset_times=times_data, dataset_vel=velocities)

        print(epoch+1, "Training Loss:", loss)


    ############## TEST ##############
    ## Generate a sequence of times ##
    ## Plot the model predictions   ##
    ## Plot the true dynamic curve  ##
    ## Plot the dataset of (t,v)    ##
    ##################################
    ## Data to plot ##
    with torch.autograd.no_grad():
        t_test = np.arange(0, 2*tau_, 1/200).reshape(800, 1)
        v_true = np.array([2 * np.exp(-t / tau_) + 4 for t in t_test])
        ## TODO: predict the velocities from t_test in order to plot them
        ## use .detach().numpy() method to convert torch.tensor to numpy
        v_test = net(torch.tensor(t_test, dtype = torch.float32)).detach().numpy()
        
        
        

    ## Plotting ##
    fig, ax = plt.subplots()
    ax.plot(t_test, v_test, label="Model")
    ax.plot(t_test, v_true, '--', label="Ground Truth")
    ax.plot(t_dataset, v_dataset, 'o', label="Dataset")

    ax.legend(loc='upper right', shadow=True)
    ax.set_xlabel('Time (s)', weight='bold')
    ax.set_ylabel('Velocity (m/s)', weight='bold')
    ax.set_title('Stokes Dynamics', fontsize=22, weight='bold')
    plt.show()


if __name__=="__main__":
    main()




