import sys
import multiprocessing
import gym
import torch
import utils
from Algorithm import PPO
 
def worker(hyperParameters,updateNetworkWeightBarrier,updateStateRespositoryEvent,stateRepositoryQueue ,policyStateDict):

    env = gym.make(hyperParameters.enviornmentName)
    stateDimension = env.observation_space.shape[0]
    actionDimension = env.action_space.shape[0]
    workerStateRepository = utils.StateRepository()
    
    actorCritic = PPO.ActorCritic(hyperParameters, stateDimension, actionDimension)
    actorCritic.load_state_dict(policyStateDict)   
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    for i_episode in range(1, hyperParameters.maxEpisodes+1):
        state = env.reset()
        for t in range(hyperParameters.maxTimesteps):
            try:
                timestep += 1
                # Running policy_old:
                state = state.reshape(1, -1)
                action, logprob = actorCritic.selectAction(state)

                workerStateRepository.states.append(state.reshape(-1))
                workerStateRepository.actions.append(action.reshape(-1))
                workerStateRepository.logprobs.append(logprob.reshape(-1))
                
                state, reward, done,_ = env.step(action.flatten())
                
                # Saving reward and statusTerminals:
                workerStateRepository.rewards.append(reward)
                workerStateRepository.terminalStatus.append(done)
                
                # update if its time
                if timestep % hyperParameters.updateTimestep == 0:
                    stateRepositoryQueue.put(workerStateRepository)
                    updateNetworkWeightBarrier.wait()
                    
                    updateStateRespositoryEvent.wait()
                    actorCritic.load_state_dict(policyStateDict)
                    workerStateRepository.clear()
                    timestep = 0
                    
                if hyperParameters.render:
                    env.render()
                if done:
                    break
                    
            except KeyboardInterrupt:
                env.close()
                sys.exit(0)
    
    env.close()
