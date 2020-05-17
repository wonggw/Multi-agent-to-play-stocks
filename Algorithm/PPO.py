import numpy
import torch
import torch.nn as nn

class ActorCriticDiscrete(nn.Module):
    def __init__(self, hyperParameters):
        super(ActorCriticDiscrete, self).__init__()
        
        self.__device = hyperParameters.device
        # actor mean range -1 to 1
        self.__actorNet = nn.Sequential(
                nn.Linear(hyperParameters.stateDimension, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, hyperParameters.actionDimension),
                nn.Softmax(dim=-1)
                ).to(self.__device)
        
        # critic
        self.__criticNet = nn.Sequential(
                nn.Linear(hyperParameters.stateDimension, 256),
                nn.Tanh(),
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
                ).to(self.__device) 
  
    def forward(self):
        raise NotImplementedError   

    def setStateDict(self,actorCriticStateDict):
        if isinstance(actorCriticStateDict,dict):
            tensorDict={}
            for key, value in actorCriticStateDict.items():
                tensorDict[key] = torch.from_numpy(value).to(self.__device)
            actorCriticStateDict =  tensorDict
  
        elif isinstance(actorCriticStateDict,str):
            actorCriticStateDict = torch.load(actorCriticStateDict)
       
        else:
            raise TypeError("actorCriticStateDict msut be a dictionary or a string!") 
        self.load_state_dict(actorCriticStateDict)
  
    def selectAction(self, state):
        state = torch.FloatTensor(state).to(self.__device)
        actionProbs = self.__actorNet(state)
        actionDistribution = torch.distributions.Categorical(actionProbs)
        action = actionDistribution.sample()
        
        return action.detach().cpu().numpy(), actionDistribution.log_prob(action).detach().cpu().numpy()
    
    def evaluate(self, state, action):
        actionProbs = self.__actorNet(state)
        actionDistribution = torch.distributions.Categorical(actionProbs)

        actionLogProbs = actionDistribution.log_prob(action)
        entropyDistribution = actionDistribution.entropy()
        stateValue = self.__criticNet(state)
        
        return actionLogProbs, torch.squeeze(stateValue), entropyDistribution
  
class ActorCriticContinuous(nn.Module):
    def __init__(self, hyperParameters):
        super(ActorCriticContinuous, self).__init__()
        
        self.__device = hyperParameters.device
        # actor mean range -1 to 1
        self.__actorNet = nn.Sequential(
                nn.Linear(hyperParameters.stateDimension, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, hyperParameters.actionDimension),
                nn.LeakyReLU()
                ).to(self.__device)
        
        # critic
        self.__criticNet = nn.Sequential(
                nn.Linear(hyperParameters.stateDimension, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1)
                ).to(self.__device)
                
        self.__actionVariance = torch.full((hyperParameters.actionDimension,), hyperParameters.actionSTD*hyperParameters.actionSTD).to(self.__device)
        
    def forward(self):
        raise NotImplementedError
   
    def setStateDict(self,actorCriticStateDict):
        if isinstance(actorCriticStateDict,dict):
            tensorDict={}
            for key, value in actorCriticStateDict.items():
                tensorDict[key] = torch.from_numpy(value).to(self.__device)
            actorCriticStateDict =  tensorDict
  
        elif isinstance(actorCriticStateDict,str):
            actorCriticStateDict = torch.load(actorCriticStateDict)
       
        else:
            raise TypeError("actorCriticStateDict msut be a dictionary or a string!") 
        self.load_state_dict(actorCriticStateDict)
  
    def selectAction(self, state):
        state = torch.FloatTensor(state).to(self.__device)
        actionMean = self.__actorNet(state)
        actionVarianceMatrix = torch.diag(self.__actionVariance).to(self.__device)
        actionDistribution = torch.distributions.MultivariateNormal(actionMean,actionVarianceMatrix)
        action = actionDistribution.sample()
        
        return action.detach().cpu().numpy(), actionDistribution.log_prob(action).detach().cpu().numpy()
    
    def evaluate(self, state, action):
        actionMean = self.__actorNet(state)
        actionVariance = self.__actionVariance.expand_as(actionMean)
        actionVarianceMatrix = torch.diag_embed(actionVariance).to(self.__device)
        actionDistribution = torch.distributions.MultivariateNormal(actionMean, actionVarianceMatrix)

        actionLogProbs = actionDistribution.log_prob(action)
        entropyDistribution = actionDistribution.entropy()
        stateValue = self.__criticNet(state)
        
        return actionLogProbs, torch.squeeze(stateValue), entropyDistribution

class PPO:
    def __init__(self, hyperParameters):
        self.__device=hyperParameters.device
        self.__epsilonClip = hyperParameters.epsilonClip
        self.__updateEpochs = hyperParameters.updateEpochs      
        self.__gamma = hyperParameters.gamma
        
        self.policyAlgorithm = ActorCriticContinuous if hyperParameters.actionContinuous else ActorCriticDiscrete
        self.actorCritic = self.policyAlgorithm(hyperParameters)
        self.__optimizer = torch.optim.Adam(self.actorCritic.parameters(), lr=hyperParameters.lr, betas=hyperParameters.betas)
        # self.optimizer = torch.optim.SGD(self.actorCritic.parameters(), lr=hyperParameters.lr, momentum=0.5, weight_decay=0.001)
        self.__MseLoss = nn.MSELoss()
  
    def __convertListToTensor(self,list):
        numpyArray = numpy.stack(list)
        arrayShape = numpyArray.shape 
        return torch.from_numpy(numpyArray.reshape(-1)).reshape(arrayShape).float().to(self.__device)

    def getStateDict(self):
        return {key:value.cpu().numpy() for key, value in self.actorCritic.state_dict().items()}
 
    def saveModel(self,directory):
        torch.save(self.actorCritic.state_dict(), directory) 
  
    def update(self, stateRepository):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, terminalStatus in zip(reversed(stateRepository.rewards), reversed(stateRepository.terminalStatus)):
            if terminalStatus:
                discounted_reward = 0
            discounted_reward = reward + (self.__gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).float().to(self.__device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        with torch.no_grad():
            statesOld = self.__convertListToTensor(stateRepository.states)
            actionsOld = self.__convertListToTensor(stateRepository.actions)
            logProbsOld = self.__convertListToTensor(stateRepository.actionLogProbs)

        # Optimize actorCritic for K epochs:
        for _ in range(self.__updateEpochs):
            # Evaluating old actions and values :
            actionLogProbs, stateValues, entropyDistribution = self.actorCritic.evaluate(statesOld, actionsOld)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(actionLogProbs - logProbsOld.detach())
  
            # Finding Surrogate Loss:
            advantages = rewards - stateValues.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.__epsilonClip, 1+self.__epsilonClip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.__MseLoss(stateValues, rewards) - 0.01*entropyDistribution
            
            # take gradient step
            self.__optimizer.zero_grad()
            loss.mean().backward()
            self.__optimizer.step()
