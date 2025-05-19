import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size= state_size
        self.action_size= action_size
        self.memory= deque(maxlen=10000)
        self.gamma= 0.95    # discount rate
        self.epsilon= 1.0   # exploration rate
        self.epsilon_min= 0.01
        self.epsilon_decay= 0.995
        self.learning_rate= 0.001
        self.model= self._build_model()
        self.target_model= self._build_model()
        self.update_target_model()
    

    #dqn with 3 hidden layers, relu activation function
    def _build_model(self):
        model= nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size))
        return model
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    #store the transition state in replay buffer
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    #epsilon-greedy policy
    def act(self, state):
        if np.random.rand()<=self.epsilon:
            return random.randrange(self.action_size)
        state=torch.FloatTensor(state)
        q_values=self.model(state)
        return torch.argmax(q_values).item() 
   
   #train if buffer has enough samples
    def replay(self, batch_size):
        if len(self.memory)<batch_size:
            return
        
        batch= random.sample(self.memory, batch_size)
        states= torch.FloatTensor(np.array([t[0] for t in batch]))
        actions= torch.LongTensor(np.array([t[1] for t in batch]))
        rewards= torch.FloatTensor(np.array([t[2] for t in batch]))
        next_states= torch.FloatTensor(np.array([t[3] for t in batch]))
        dones= torch.FloatTensor(np.array([t[4] for t in batch]))
        
        #compute Q-values
        current_q= self.model(states).gather(1, actions.unsqueeze(1))   #picks the Q-value corresponding to the action that was actually taken
        #compute target Q-values
        next_q= self.target_model(next_states).max(1)[0].detach()   #takes the maximum Q-value over all possible actions for each next state
        target= rewards+(1-dones)*self.gamma*next_q                 #zero out the future reward if the episode terminated at this step
        
        #compute loss
        loss= nn.MSELoss()(current_q.squeeze(), target)

        #gradient step
        optimizer= optim.Adam(self.model.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #epsilon decay
        if self.epsilon> self.epsilon_min:
            self.epsilon*= self.epsilon_decay

    def save_model(self,path):
        torch.save(self.model.state_dict(),path)

    def load_model(self,path):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.model.state_dict())
