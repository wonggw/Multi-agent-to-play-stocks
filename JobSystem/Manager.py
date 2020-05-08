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
    policyStateDict = multiprocessing.Manager().dict()
    policyStateDict = ppo.getPolicyStateDict()
    stateRepositoryQueue = multiprocessing.Manager().Queue() 
    stateRepository = utils.StateRepository()

    updateNetworkWeightBarrier = multiprocessing.Barrier(hyperParameters.processNumber+1) 
    updateStateRespositoryEvent = multiprocessing.Event()
 
    processes=[multiprocessing.Process(target=Worker.worker , args=(hyperParameters,updateNetworkWeightBarrier,updateStateRespositoryEvent,stateRepositoryQueue ,policyStateDict)) for _ in range(hyperParameters.processNumber)]
    for process in processes:
        process.start()

    # logging variables
    running_reward = 0
    avg_length = 0
    learnStep = 0

    while True:

        stateRepository.clear()
        updateNetworkWeightBarrier.wait()

        learnStep+=1
        
        for _ in range(hyperParameters.processNumber):
            stateRepository.add(stateRepositoryQueue.get())
        
        ppo.update(stateRepository)
        
        policyStateDict = ppo.getPolicyStateDict()
        updateStateRespositoryEvent.set()
        
        if (learnStep%hyperParameters.logInterval==0):
          print('Episode {} \t avg length: {} \t reward: {}'.format(learnStep, len(stateRepository.states)/hyperParameters.processNumber, sum(stateRepository.rewards)/hyperParameters.processNumber))
        
       # save every 500 episodes
        if (learnStep % 100 == 0):
            torch.save(ppo.policy.state_dict(), './PPO_multiagent_{}.pth'.format(hyperParameters.enviornmentName)) 
        
    for process in processes:
        process.join()
