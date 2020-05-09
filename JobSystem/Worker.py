import sys
import multiprocessing
import gym
import torch
import utils
import numpy
from Algorithm import PPO
 
def worker(hyperParameters,updateNetworkWeightBarrier,stateRepositoryQueue ,policyStateDict,policyStateDictQueue ,processID):

    env = gym.make(hyperParameters.enviornmentName)
    stateDimension = env.observation_space.shape[0]
    actionDimension = env.action_space.shape[0]
    workerStateRepository = utils.StateRepository()
    
    if (hyperParameters.actionContinuous):
        actorCritic = PPO.ActorCriticContinuous(hyperParameters, stateDimension, actionDimension)
    else:
        actorCritic = PPO.ActorCriticDiscrete(hyperParameters, stateDimension, actionDimension)
   
    actorCritic.load_state_dict(actorCritic.convertNumpyToTensorDict(policyStateDict))   
    
    # logging variables
    runningRewards = 0
    averageSteps = 0
    timeStep = 0
    
    for episode in range(1, hyperParameters.maxEpisodes+1):
        state = env.reset()
        for t in range(hyperParameters.maxTimesteps):
            try:
                timeStep += 1
                # Running policy_old:
                action, logprob = actorCritic.selectAction(state)
                workerStateRepository.states.append(numpy.array(state))
                workerStateRepository.actions.append(action)
                workerStateRepository.logprobs.append(logprob)
                
                if (hyperParameters.actionContinuous):
                    state, reward, done,_ = env.step(action)
                else:
                    state, reward, done,_ = env.step(action.item())

                # Saving reward and statusTerminals:
                workerStateRepository.rewards.append(reward)
                workerStateRepository.terminalStatus.append(done)
                
                runningRewards += reward
                # update if its time
                if timeStep % hyperParameters.updateTimestep == 0:
                    stateRepositoryQueue.put(workerStateRepository)
                    updateNetworkWeightBarrier.wait()
                    
                    actorCriticStateDict=actorCritic.convertNumpyToTensorDict(policyStateDictQueue.get())
                    actorCritic.load_state_dict(actorCriticStateDict)
                    
                    workerStateRepository.clear()
                    timeStep = 0

                if hyperParameters.render:
                    env.render()
                    
                # if processID==0 and episode% (hyperParameters.logInterval*5) == 0:  
                    # env.render()
                    
                if done:
                    break
                    
            except KeyboardInterrupt:
                env.close()
                sys.exit(0)        # logging

        averageSteps += t

        if processID==0 and episode% hyperParameters.logInterval == 0:
            averageSteps = int(averageSteps/hyperParameters.logInterval)
            runningRewards = int(runningRewards/hyperParameters.logInterval)
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(episode, averageSteps, runningRewards))
            runningRewards = 0
            averageSteps = 0
    
    env.close()
