import torch

class HyperParameters:
    def __init__(self):
    
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.numberOfWorkers = 2
        
        self.enviornmentName = "BipedalWalker-v3"
        self.stateDimension = 0
        self.actionDimension = 0
        self.actionContinuous  = True
        # self.render = True
        self.render = False
        
        self.logInterval = 1                # print avg reward in the interval
        self.solvedReward = 300             # stop training if averageReward > solvedReward
        
        self.maxEpisodes = 10000000         # max training episodes
        self.maxTimesteps = 2000           # max timesteps in one episode
        self.updateTimestep = 500          # Update policy every n timesteps
        self.updateEpochs = 50              # Update policy for K epochs
        
        self.actionSTD = 0.5               # constant std for action distribution (Multivariate Normal)
        self.epsilonClip = 0.2              # clip parameter for PPO
        self.gamma = 0.99                   # discount factor
        
        self.lr = 0.0003
        self.gamma = 0.99
        self.betas = (0.9, 0.999)
        
        self.random_seed = None
  
class StateRepository:
    def __init__(self):
        self.states = []
        self.actions = []
        self.actionLogProbs = []
        self.rewards = []
        self.terminalStatus = []

    def add(self, newStateRepository):
        self.states +=  newStateRepository.states
        self.actions +=  newStateRepository.actions
        self.actionLogProbs +=  newStateRepository.actionLogProbs
        self.rewards +=  newStateRepository.rewards
        self.terminalStatus +=  newStateRepository.terminalStatus
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.actionLogProbs[:]
        del self.rewards[:]
        del self.terminalStatus[:] 