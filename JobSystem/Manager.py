import copy
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
    
    managerActions= multiprocessing.Manager().list()
    managerStates= multiprocessing.Manager().list()
    managerLogprobs= multiprocessing.Manager().list()
    managerRewards= multiprocessing.Manager().list()
    managerTerminalStatus= multiprocessing.Manager().list()
    stateRepository = utils.StateRepository()
    
    updateNetworkWeightBarrier = multiprocessing.Barrier(hyperParameters.processNumber+1) 
    updateStateRespositoryEvent = multiprocessing.Event()
    
    processes=[multiprocessing.Process(target=Worker.worker , args=(hyperParameters,updateNetworkWeightBarrier,updateStateRespositoryEvent,managerActions,managerStates,managerLogprobs,managerRewards,managerTerminalStatus,policyStateDict)) for _ in range(hyperParameters.processNumber)]
    for process in processes:
        process.start()

    # logging variables
    running_reward = 0
    avg_length = 0
    learnStep = 0

    while True:

        updateNetworkWeightBarrier.wait()

        learnStep+=1
        
        stateRepository.actions =  list(copy.copy(managerActions))
        stateRepository.states =  list(copy.copy(managerStates))
        stateRepository.logprobs = list(copy.copy(managerLogprobs))
        stateRepository.rewards =  list(copy.copy(managerRewards))
        stateRepository.terminalStatus =  list(copy.copy(managerTerminalStatus))
        
        ppo.update(stateRepository)
        
        policyStateDict = ppo.getPolicyStateDict()
        updateStateRespositoryEvent.set()
        
        del managerActions[:]
        del managerStates[:]
        del managerLogprobs[:]
        del managerRewards[:]
        del managerTerminalStatus[:]
        
        if (learnStep%hyperParameters.logInterval==0):
          print('Episode {} \t avg length: {} \t reward: {}'.format(learnStep, len(stateRepository.states)/hyperParameters.processNumber, sum(stateRepository.rewards)/hyperParameters.processNumber))
        
       # save every 500 episodes
        if (learnStep % 100 == 0):
            torch.save(ppo.policy.state_dict(), './PPO_multiagent_{}.pth'.format(hyperParameters.enviornmentName)) 
        
    for process in processes:
        process.join()
