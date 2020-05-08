import numpy
import gym
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class ActorCritic(nn.Module):
    def __init__(self, hyperParameters,stateDimension, actionDimension):
        super(ActorCritic, self).__init__()
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
        
    def selectAction(self, state):
        actionMean = self.actorNet(torch.FloatTensor(state).to(self.device))
        actionVarianceMatrix = torch.diag(self.actionVariance).to(self.device)
        actionDistribution = MultivariateNormal(actionMean,actionVarianceMatrix)
        action = actionDistribution.sample()
        
        return action.detach().cpu().data.numpy() , actionDistribution.log_prob(action).detach().cpu().data.numpy()
    
    def evaluate(self, state, action):
        actionMean = self.actorNet(state)
        actionVariance = self.actionVariance.expand_as(actionMean)
        actionVarianceMatrix = torch.diag_embed(actionVariance).to(self.device)
        actionDistribution = MultivariateNormal(actionMean, actionVarianceMatrix)

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
        
        self.rewards=torch.tensor(0).float().to(self.device)
        self.statesOld = 0
        self.actionsOld = 0
        self.logProbsOld = 0      
 
        self.policy = ActorCritic(hyperParameters, stateDimension, actionDimension).to(hyperParameters.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.0003 , betas=(0.9, 0.999))
        # self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=hyperParameters.lr, momentum=0.2,dampening=0.1, weight_decay=0.01)
        self.MseLoss = nn.MSELoss()
  
    def convertListToTensor(self,list):
        numpyArray = numpy.array(list)
        arrayShape = numpyArray.shape 
        return torch.from_numpy(numpyArray.reshape(-1)).reshape(arrayShape).float().to(self.device).detach()

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
        
        self.convertListToTensor(stateRepository.states)
        # convert list to tensor
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
        return {key:value.to('cpu') for key, value in self.policy.state_dict().items()}


def main():
    ############## Hyperparameters ##############
    env_name = "BipedalWalker-v3"
    # render = True
    render = False
    solved_reward = 300         # stop training if avg_reward > solved_reward
    log_interval = 10           # print avg reward in the interval
    
    max_episodes = 10000        # max training episodes
    maxTimesteps = 1500        # max timesteps in one episode
    updateTimestep = 1500     # Update policy every n timesteps
    updateEpochs = 50               # Update policy for K epochs
    
    actionSTD = 0.5            # constant std for action distribution (Multivariate Normal)
    epsilonClip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.01
    
    random_seed = None
    #############################################
    
    env = gym.make(env_name)
    stateDimension = env.observation_space.shape[0]
    actionDimension = env.action_space.shape[0]

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    
    stateRepository = StateRepository()
    ppo = PPO(stateDimension, actionDimension, actionSTD, lr, gamma, updateEpochs, epsilonClip)
    print(lr)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(maxTimesteps):
            timestep += 1
            
            # Running policy_old:
            action = ppo.policyOld.selectAction(state, stateRepository)
            state, reward, done,_ = env.step(action)
            
            # Saving reward and statusTerminals:
            stateRepository.rewards.append(reward)
            stateRepository.statusTerminals.append(done)
            
            running_reward += reward
            
            # update if its time
            if timestep % updateTimestep == 0:
                ppo.update(stateRepository)
                stateRepository.clear()
                timestep = 0
            
            if render:
                env.render()
            if done:
                break
                
        avg_length += t
        
        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))
            break
            
        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
  
# if __name__ == '__main__':
    # main()
    
