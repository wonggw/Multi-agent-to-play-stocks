import sys
import numpy
import gym

import utils
import Enviornment
 
def ActtionDiscreteToContinuous(actionDiscrete):
    actionContinuous=numpy.zeros(4)
    if (actionDiscrete==0):
        return actionContinuous
    else:
        index=((actionDiscrete-1)//2)
        if (actionDiscrete%2==1):
            value=-0.5
        else:
            value=0.5
        actionContinuous[index]=value
        return actionContinuous

class EnviornmentSetup():
    def __init__(self, hyperParameters,policyAlgorithm):
        # self.env = gym.make(hyperParameters.enviornmentName)
        self.env = Enviornment.StockTradingEnvironment()
        self.actionContinuous = hyperParameters.actionContinuous
        self.stateRepository = utils.StateRepository()
        self.policy = policyAlgorithm(hyperParameters)
        
        self.__state = 0
        self.__action = 0
        self.__actionLogProb = 0
        self.__reward = 0
        self.__done = False

    def selectAction(self,state):
        self.__state=state
        self.__action, self.__actionLogProb = self.policy.selectAction(self.__state)
        return self.__action
  
    def inputEnvStep(self,action):
        action = action if self.actionContinuous else action.item()
        stateNext, self.__reward, self.__done, info = self.env.step(action)
        return stateNext, self.__reward, self.__done,info
        
    def loadToStateRepository(self):
        self.stateRepository.states.append(numpy.array(self.__state))
        self.stateRepository.actions.append(self.__action)
        self.stateRepository.actionLogProbs.append(self.__actionLogProb)
        self.stateRepository.rewards.append(self.__reward)
        self.stateRepository.terminalStatus.append(self.__done)
 
    
def worker(hyperParameters,policyAlgorithm,updateNetworkWeightBarrier,stateRepositoryQueue,actorCriticStateDictQueue,actorCriticStateDict,processID):
        
    enviornmentSetup=EnviornmentSetup(hyperParameters,policyAlgorithm)
    enviornmentSetup.policy.stateDict = actorCriticStateDict
    
    # logging variables
    runningRewards = 0
    averageSteps = 0
    timeStep = 0
    
    for episode in range(1, hyperParameters.maxEpisodes+1):
        state = enviornmentSetup.env.reset()
        if processID==0:
            print(enviornmentSetup.env.ticker)
        for t in range(hyperParameters.maxTimesteps):
            try:
                timeStep += 1

                action = enviornmentSetup.selectAction(state)
                state, reward, done,_= enviornmentSetup.inputEnvStep(action)
                enviornmentSetup.loadToStateRepository()
                runningRewards += reward
                
                # update if its time
                if timeStep % hyperParameters.updateTimestep == 0:
                    stateRepositoryQueue.put(enviornmentSetup.stateRepository)
                    updateNetworkWeightBarrier.wait()
  
                    enviornmentSetup.policy.setStateDict(actorCriticStateDictQueue.get())
                    enviornmentSetup.stateRepository.clear()
                    timeStep = 0

                if hyperParameters.render:
                    enviornmentSetup.env.render()
                    
                # if processID==0 and episode% (hyperParameters.logInterval*5) == 0:  
                    # env.render()
                    
                if done:
                    break
                    
            except KeyboardInterrupt:
                enviornmentSetup.env.close()
                sys.exit(0)        # logging

        averageSteps += t

        if processID==0 and episode% hyperParameters.logInterval == 0:
            averageSteps = int(averageSteps/hyperParameters.logInterval)
            runningRewards = int(runningRewards/hyperParameters.logInterval)
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(episode, averageSteps, runningRewards))
            runningRewards = 0
            averageSteps = 0
    
    enviornmentSetup.env.close()
    