import numpy
import torch
import torch.nn as nn

class ActorCriticDiscrete(nn.Module):
    def __init__(self, hyperParameters):
        super(ActorCriticDiscrete, self).__init__()
        
        self.device = hyperParameters.device
        # actor mean range -1 to 1
        self.actorNet = nn.Sequential(
                nn.Linear(hyperParameters.stateDimension, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, hyperParameters.actionDimension),
                nn.Softmax(dim=-1)
                ).to(self.device)
        
        # critic
        self.criticNet = nn.Sequential(
                nn.Linear(hyperParameters.stateDimension, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
                ).to(self.device) 
  
    def forward(self):
        raise NotImplementedError   

    def setStateDict(self,actorCriticStateDict):
        if isinstance(actorCriticStateDict,dict):
            tensorDict={}
            for key, value in actorCriticStateDict.items():
                tensorDict[key] = torch.from_numpy(value).to(self.device)
            actorCriticStateDict =  tensorDict
  
        elif isinstance(actorCriticStateDict,str):
            actorCriticStateDict = torch.load(actorCriticStateDict)
       
        else:
            raise TypeError("actorCriticStateDict msut be a dictionary or a string!") 
        self.load_state_dict(actorCriticStateDict)
  
    def selectAction(self, state):
        state = torch.FloatTensor(state).to(self.device)
        actionProbs = self.actorNet(state)
        actionDistribution = torch.distributions.Categorical(actionProbs)
        action = actionDistribution.sample()
        
        return action.detach().cpu().numpy(), actionDistribution.log_prob(action).detach().cpu().numpy()
    
    def evaluate(self, state, action):
        actionProbs = self.actorNet(state)
        actionDistribution = torch.distributions.Categorical(actionProbs)

        actionLogProbs = actionDistribution.log_prob(action)
        entropyDistribution = actionDistribution.entropy()
        stateValue = self.criticNet(state)
        
        return actionLogProbs, torch.squeeze(stateValue), entropyDistribution
  
class ActorCriticContinuous(nn.Module):
    def __init__(self, hyperParameters):
        super(ActorCriticContinuous, self).__init__()
        
        self.device = hyperParameters.device
        # actor mean range -1 to 1
        self.actorNet = nn.Sequential(
                nn.Linear(hyperParameters.stateDimension, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, hyperParameters.actionDimension),
                nn.Tanh()
                ).to(self.device)
        
        # critic
        self.criticNet = nn.Sequential(
                nn.Linear(hyperParameters.stateDimension, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                ).to(self.device)
                
        self.actionVariance = torch.full((hyperParameters.actionDimension,), hyperParameters.actionSTD*hyperParameters.actionSTD).to(self.device)
        
    def forward(self):
        raise NotImplementedError
   
    def setStateDict(self,actorCriticStateDict):
        if isinstance(actorCriticStateDict,dict):
            tensorDict={}
            for key, value in actorCriticStateDict.items():
                tensorDict[key] = torch.from_numpy(value).to(self.device)
            actorCriticStateDict =  tensorDict
  
        elif isinstance(actorCriticStateDict,str):
            actorCriticStateDict = torch.load(actorCriticStateDict)
       
        else:
            raise TypeError("actorCriticStateDict msut be a dictionary or a string!") 
        self.load_state_dict(actorCriticStateDict)
  
    def selectAction(self, state):
        state = torch.FloatTensor(state).to(self.device)
        actionMean = self.actorNet(state)
        actionVarianceMatrix = torch.diag(self.actionVariance).to(self.device)
        actionDistribution = torch.distributions.MultivariateNormal(actionMean,actionVarianceMatrix)
        action = actionDistribution.sample()
        
        return action.detach().cpu().numpy(), actionDistribution.log_prob(action).detach().cpu().numpy()
    
    def evaluate(self, state, action):
        actionMean = self.actorNet(state)
        actionVariance = self.actionVariance.expand_as(actionMean)
        actionVarianceMatrix = torch.diag_embed(actionVariance).to(self.device)
        actionDistribution = torch.distributions.MultivariateNormal(actionMean, actionVarianceMatrix)

        actionLogProbs = actionDistribution.log_prob(action)
        entropyDistribution = actionDistribution.entropy()
        stateValue = self.criticNet(state)
        
        return actionLogProbs, torch.squeeze(stateValue), entropyDistribution

class PPO:
    def __init__(self, hyperParameters):
        self.device=hyperParameters.device
        self.lr = hyperParameters.lr
        self.gamma = hyperParameters.gamma
        self.epsilonClip = hyperParameters.epsilonClip
        self.updateEpochs = hyperParameters.updateEpochs      
 
        if (hyperParameters.actionContinuous):
            self.policyAlgorithm = ActorCriticContinuous
        else:
            self.policyAlgorithm = ActorCriticDiscrete

        self.actorCritic = self.policyAlgorithm(hyperParameters)
        self.optimizer = torch.optim.Adam(self.actorCritic.parameters(), lr=hyperParameters.lr, betas=hyperParameters.betas)
        # self.optimizer = torch.optim.SGD(self.actorCritic.parameters(), lr=hyperParameters.lr, momentum=0.5, weight_decay=0.001)
        self.MseLoss = nn.MSELoss()
  
    def __convertListToTensor(self,list):
        numpyArray = numpy.stack(list)
        arrayShape = numpyArray.shape 
        return torch.from_numpy(numpyArray.reshape(-1)).reshape(arrayShape).float().to(self.device)

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
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).float().to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        with torch.no_grad():
            statesOld = self.__convertListToTensor(stateRepository.states)
            actionsOld = self.__convertListToTensor(stateRepository.actions)
            logProbsOld = self.__convertListToTensor(stateRepository.actionLogProbs)

        # Optimize actorCritic for K epochs:
        for _ in range(self.updateEpochs):
            # Evaluating old actions and values :
            actionLogProbs, stateValues, entropyDistribution = self.actorCritic.evaluate(statesOld, actionsOld)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(actionLogProbs - logProbsOld.detach())
  
            # Finding Surrogate Loss:
            advantages = rewards - stateValues.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.epsilonClip, 1+self.epsilonClip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(stateValues, rewards) - 0.01*entropyDistribution
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
