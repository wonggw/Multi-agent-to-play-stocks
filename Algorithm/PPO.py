import numpy
import gym
import torch
import torch.nn as nn

class ActorCriticDiscrete(nn.Module):
    def __init__(self, hyperParameters,stateDimension, actionDimension):
        super(ActorCriticDiscrete, self).__init__()
        self.device = hyperParameters.device
        # actor mean range -1 to 1
        self.actorNet = nn.Sequential(
                nn.Linear(stateDimension, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, actionDimension),
                nn.Softmax(dim=-1)
                ).to(self.device)
        
        # critic
        self.criticNet = nn.Sequential(
                nn.Linear(stateDimension, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
                ).to(self.device) 
  
    def forward(self):
        raise NotImplementedError
        
    def convertNumpyToTensorDict(self,numpyDict):
        tensorDict={}
        for key, value in numpyDict.items():
            tensorDict[key] = torch.from_numpy(value).to(self.device)
        return tensorDict

    def selectAction(self, state):
        state = torch.FloatTensor(state).to(self.device)
        actionProbs = self.actorNet(state)
        actionDistribution = torch.distributions.Categorical(actionProbs)
        action = actionDistribution.sample()
        
        return action.detach().cpu().numpy(), actionDistribution.log_prob(action).detach().cpu().numpy()
    
    def evaluate(self, state, action):
        actionProbs = self.actorNet(state)
        actionDistribution = torch.distributions.Categorical(actionProbs)

        action_logprobs = actionDistribution.log_prob(action)
        entropyDistribution = actionDistribution.entropy()
        stateValue = self.criticNet(state)
        
        return action_logprobs, torch.squeeze(stateValue), entropyDistribution
  
class ActorCriticContinuous(nn.Module):
    def __init__(self, hyperParameters,stateDimension, actionDimension):
        super(ActorCriticContinuous, self).__init__()
        self.device = hyperParameters.device
        # actor mean range -1 to 1
        self.actorNet = nn.Sequential(
                nn.Linear(stateDimension, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, actionDimension),
                nn.Tanh()
                ).to(self.device)
        
        # critic
        self.criticNet = nn.Sequential(
                nn.Linear(stateDimension, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                ).to(self.device)
                
        self.actionVariance = torch.full((actionDimension,), hyperParameters.actionSTD*hyperParameters.actionSTD).to(self.device)
        
    def forward(self):
        raise NotImplementedError
        
    def convertNumpyToTensorDict(self,numpyDict):
        tensorDict={}
        for key, value in numpyDict.items():
            tensorDict[key] = torch.from_numpy(value).to(self.device)
        return tensorDict
    
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

        action_logprobs = actionDistribution.log_prob(action)
        entropyDistribution = actionDistribution.entropy()
        stateValue = self.criticNet(state)
        
        return action_logprobs, torch.squeeze(stateValue), entropyDistribution

class PPO:
    def __init__(self, stateDimension, actionDimension, hyperParameters):
        self.device=hyperParameters.device
        self.lr = hyperParameters.lr
        self.gamma = hyperParameters.gamma
        self.epsilonClip = hyperParameters.epsilonClip
        self.updateEpochs = hyperParameters.updateEpochs      
 
        if (hyperParameters.actionContinuous):
            self.policy = ActorCriticContinuous(hyperParameters, stateDimension, actionDimension)
        else:
            self.policy = ActorCriticDiscrete(hyperParameters, stateDimension, actionDimension)
            
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=hyperParameters.lr, betas=(0.9, 0.999))
        # self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=hyperParameters.lr, momentum=0.2,dampening=0.1, weight_decay=0.01)
        self.MseLoss = nn.MSELoss()
  
    def convertListToTensor(self,list):
        numpyArray = numpy.stack(list)
        arrayShape = numpyArray.shape 
        return torch.from_numpy(numpyArray.reshape(-1)).reshape(arrayShape).float().to(self.device)

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
            statesOld = self.convertListToTensor(stateRepository.states)
            actionsOld = self.convertListToTensor(stateRepository.actions)
            logProbsOld = self.convertListToTensor(stateRepository.logprobs)

        # Optimize policy for K epochs:
        for _ in range(self.updateEpochs):
            # Evaluating old actions and values :
            logprobs, stateValues, dist_entropy = self.policy.evaluate(statesOld, actionsOld)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - logProbsOld.detach())
  
            # Finding Surrogate Loss:
            advantages = rewards - stateValues.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.epsilonClip, 1+self.epsilonClip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(stateValues, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
   
    def getPolicyStateDict(self):
        return {key:value.cpu().numpy() for key, value in self.policy.state_dict().items()}
  