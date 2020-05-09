import torch
import numpy as np

class HyperParameters:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.numberOfWorkers = 4
        self.enviornmentName = "BipedalWalker-v3"
        self.actionContinuous  = True
        # self.render = True
        self.render = False
        self.solved_reward = 300            # stop training if avg_reward > solved_reward
        self.logInterval = 2               # print avg reward in the interval
        
        self.maxEpisodes = 10000000            # max training episodes
        self.maxTimesteps = 1599            # max timesteps in one episode
        self.updateTimestep = 1000           # Update policy every n timesteps
        self.updateEpochs = 80              # Update policy for K epochs
        
        self.actionSTD = 0.5                # constant std for action distribution (Multivariate Normal)
        self.epsilonClip = 0.2              # clip parameter for PPO
        self.gamma = 0.99                   # discount factor
        
        self.lr = 0.0003 
        
        self.random_seed = None
  
class StateRepository:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.terminalStatus = []

    def add(self, newStateRepository):
        self.actions +=  newStateRepository.actions
        self.states +=  newStateRepository.states
        self.logprobs +=  newStateRepository.logprobs
        self.rewards +=  newStateRepository.rewards
        self.terminalStatus +=  newStateRepository.terminalStatus
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.terminalStatus[:] 