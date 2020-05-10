import gym

import utils
from Algorithm import PPO
from JobSystem import Worker

def test():
    hyperParameters = utils.HyperParameters()
    directory='./TrainedModel/PPO_multiagent_{}.pth'.format(hyperParameters.enviornmentName)
    
    env = gym.make(hyperParameters.enviornmentName)
    hyperParameters.stateDimension = env.observation_space.shape[0]
    hyperParameters.actionDimension = env.action_space.shape[0]
    
    ppo = PPO.PPO(hyperParameters)
    enviornmentSetup=Worker.EnviornmentSetup(hyperParameters,ppo.policyAlgorithm)
    enviornmentSetup.policy.setStateDict(directory)

    for episode in range(1, hyperParameters.maxEpisodes+1):
        state = enviornmentSetup.env.reset()
        while True:
     
            action = enviornmentSetup.selectAction(state)
            state, reward, done,_= enviornmentSetup.inputEnvStep(action)
            enviornmentSetup.env.render()
            
            if done:
                break
    

if __name__ == '__main__':
    test()