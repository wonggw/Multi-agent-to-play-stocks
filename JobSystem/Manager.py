import multiprocessing
import gym
import torch
import utils
from Algorithm import PPO
from JobSystem import Worker

def manager():

    hyperParameters = utils.HyperParameters()
    env = gym.make(hyperParameters.enviornmentName)
    stateDimension = env.observation_space.shape[0]
    actionDimension = env.action_space.shape[0]
    env.close()
    
    ppo = PPO.PPO(stateDimension, actionDimension, hyperParameters)
    stateRepositoryQueue = multiprocessing.Queue() 
    stateRepository = utils.StateRepository()
    policyStateDictQueue = multiprocessing.Queue()
    policyStateDict = ppo.getPolicyStateDict()
    
    updateNetworkWeightBarrier = multiprocessing.Barrier(hyperParameters.numberOfWorkers+1) 
 
    processes=[multiprocessing.Process(target=Worker.worker , args=(hyperParameters,updateNetworkWeightBarrier,stateRepositoryQueue ,policyStateDict,policyStateDictQueue ,processID)) for processID in range(hyperParameters.numberOfWorkers)]
    for process in processes:
        process.start()

    # logging variables
    # running_reward = 0
    # avg_length = 0
    learnStep = 0

    while True:

        stateRepository.clear()
        updateNetworkWeightBarrier.wait()

        learnStep+=1
        
        for _ in range(hyperParameters.numberOfWorkers):
            stateRepository.add(stateRepositoryQueue.get())
        ppo.update(stateRepository)
        
        policyStateDict = ppo.getPolicyStateDict()
        
        for _ in range(hyperParameters.numberOfWorkers):
            policyStateDictQueue.put(policyStateDict)
            
        
        # if (learnStep%hyperParameters.logInterval==0):
          # print('Episode {} \t avg length: {} \t reward: {}'.format(learnStep, len(stateRepository.states)/hyperParameters.numberOfWorkers, sum(stateRepository.rewards)/hyperParameters.numberOfWorkers))
        
       # save every 500 episodes
        if (learnStep % 100 == 0):
            print("~~~~~~~~~~Saving Model~~~~~~~~~~")
            torch.save(ppo.policy.state_dict(), './PPO_multiagent_{}.pth'.format(hyperParameters.enviornmentName)) 
        
    for process in processes:
        process.join()
