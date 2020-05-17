import multiprocessing
import gym

import utils
from Algorithm import PPO
from JobSystem import Worker

import Enviornment

def manager():

    hyperParameters = utils.HyperParameters()
    # env = gym.make(hyperParameters.enviornmentName)
    env = Enviornment.StockTradingEnvironment()
    # hyperParameters.stateDimension = env.observation_space.shape
    hyperParameters.stateDimension = (431)
    hyperParameters.actionDimension = env.action_space.shape[0]
    env.close()
    
    ppo = PPO.PPO(hyperParameters)

    stateRepositoryQueue = multiprocessing.Queue() 
    stateRepository = utils.StateRepository()
    actorCriticStateDictQueue = multiprocessing.Queue()
    actorCriticStateDict = ppo.getStateDict()
    
    updateNetworkWeightBarrier = multiprocessing.Barrier(hyperParameters.numberOfWorkers+1) 
 
    processes=[multiprocessing.Process(target=Worker.worker , args=(hyperParameters,ppo.policyAlgorithm,updateNetworkWeightBarrier,stateRepositoryQueue,actorCriticStateDictQueue,actorCriticStateDict ,processID)) for processID in range(hyperParameters.numberOfWorkers)]
    for process in processes:
        process.start()

    learnStep = 0

    while True:

        stateRepository.clear()
        updateNetworkWeightBarrier.wait()

        learnStep+=1
        
        for _ in range(hyperParameters.numberOfWorkers):
            stateRepository.add(stateRepositoryQueue.get())
        ppo.update(stateRepository)
        
        actorCriticStateDict = ppo.getStateDict()
        
        for _ in range(hyperParameters.numberOfWorkers):
            actorCriticStateDictQueue.put(actorCriticStateDict)  
        
        # if (learnStep%hyperParameters.logInterval==0):
          # print('Episode {} \t avg length: {} \t reward: {}'.format(learnStep, len(stateRepository.states)/hyperParameters.numberOfWorkers, sum(stateRepository.rewards)/hyperParameters.numberOfWorkers))
        
       # save every 500 episodes
        if (learnStep % 100 == 0):
            print("~~~~~~~~~~Saving Model~~~~~~~~~~")
            ppo.saveModel('./TrainedModel/PPO_multiagent_{}.pth'.format(hyperParameters.enviornmentName)) 
        
    for process in processes:
        process.join()
