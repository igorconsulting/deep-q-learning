import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    """
    A Deep Q-Network (DQN) implementation for reinforcement learning.
    
    This neural network is designed to approximate the Q-value function in
    reinforcement learning, taking in the state of the environment and outputting
    Q-values for each possible action. It consists of three fully connected layers
    with ReLU activations between them. The network uses the Adam optimizer and MSE
    loss function for training.
    """
    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_actions):
        """
        Initializes the Deep Q-Network model.

        Parameters:
        ----------
        learning_rate : float
            Learning rate for the optimizer.
        input_dims : tuple
            Dimensions of the input (state) space.
        fc1_dims : int
            Number of neurons in the first fully connected layer.
        fc2_dims : int
            Number of neurons in the second fully connected layer.
        n_actions : int
            Number of possible actions in the action space (output layer size).
        """
        super(DeepQNetwork, self).__init__()

        # Network architecture
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # Fully connected layers
        self.fc1 = nn.Linear(self.input_dims[0], self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        # Device configuration (GPU or CPU)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass of the neural network.

        Parameters:
        ----------
        state : torch.Tensor
            Input state of the environment.

        Returns:
        -------
        torch.Tensor
            Q-values for each possible action.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

    def save_checkpoint(self, filename='model.pth'):
        """
        Saves the current model weights to a file.

        Parameters:
        ----------
        filename : str
            The filename where to save the model.
        """
        T.save(self.state_dict(), filename)
    
    def load_checkpoint(self, filename='model.pth'):
        """
        Loads the model weights from a file.

        Parameters:
        ----------
        filename : str
            The filename from which to load the model.
        """
        self.load_state_dict(T.load(filename))

    def learn(self, state, action, reward, state_, done, gamma=0.99):
        """
        Trains the neural network by updating the Q-values based on the given
        state, action, reward, next state, and done flag.

        Parameters:
        ----------
        state : torch.Tensor
            Current state of the environment.
        action : int
            Action taken in the current state.
        reward : float
            Reward received after taking the action.
        state_ : torch.Tensor
            Next state of the environment.
        done : bool
            Flag indicating if the episode is done.
        gamma : float, optional
            Discount factor for future rewards, by default 0.99.
        """
        self.optimizer.zero_grad()
        states = state.float().to(self.device)
        actions = T.tensor(action).to(self.device)
        rewards = T.tensor(reward).to(self.device)
        states_ = state_.float().to(self.device)
        dones = T.tensor(done).to(self.device)

        # Forward pass through the network
        q_pred = self.forward(states)[actions]  # Q-value of the current action
        
        # Compute the target Q-value
        q_next = self.forward(states_).max(dim=1)[0]
        q_next[dones] = 0.0  # If the episode is done, no future reward
        q_target = rewards + gamma * q_next

        # Compute loss and backpropagate
        loss = self.loss(q_target, q_pred)
        loss.backward()
        self.optimizer.step()

