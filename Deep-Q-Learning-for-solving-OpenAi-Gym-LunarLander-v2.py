#WRITTEN BY MOHAMMAD ASADOLAHI mohammad.e.asadolahi@gmail.com
#https://github.com/mohammadAsadolahi

import numpy as np
import matplotlib.pyplot as plt
import gym
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
import keras
import numpy as np

!pip3 install box2d-py
!pip3 install gym[Box_2D]

env = gym.make('LunarLander-v2')

class replayBuffer:
  def __init__(self,maxSize,stateDim):
    self.state=np.zeros((maxSize,stateDim))
    self.action=np.zeros(maxSize,dtype= np.int8)
    self.reward=np.zeros(maxSize)
    self.done=np.zeros(maxSize,dtype= np.int8)
    self.nextState=np.zeros((maxSize,stateDim))
    self.maxSize=maxSize
    self.curser=0
    self.size=0

  def save(self,state,action,reward,nextState,done):
    self.state[self.curser]=state
    self.action[self.curser]=action
    self.reward[self.curser]=reward
    self.nextState[self.curser]=nextState
    self.done[self.curser]=done
    self.curser=(self.curser+1)%self.maxSize
    if self.size<self.maxSize:
      self.size+=1 
      
  def sample(self,batchSize):
    batchSize=min(self.size,batchSize-1)
    indexes=np.random.choice([i for i in range(self.size-1)],batchSize)
    return self.state[indexes],self.action[indexes],self.reward[indexes],self.nextState[indexes],self.done[indexes]

class Agent:
  def __init__(self,stateShape,actionShape,exploreRate=1.0,exploreRateDecay=0.9999,minimumExploreRate=0.01\
               ,gamma=0.99,saveAfterIterations=10000,modelName="DQN_LunarLanderV2.h"):
      self.gamma=gamma
      self.exploreRate=exploreRate
      self.exploreRateDecay=exploreRateDecay
      self.minimumExploreRate=minimumExploreRate
      self.actionShape=actionShape
      self.memory=replayBuffer(1000000,stateShape)
      self.buildModel(stateShape,actionShape)
      self.saveAfterIterations=saveAfterIterations
      self.updateIterations=0
      self.modelName=modelName
  
  def buildModel(self,input,output):
    inputLayer=keras.Input(shape=(input,))
    layer=Dense(256,activation='relu')(inputLayer)
    layer=Dense(256,activation='relu')(layer)
    outputLayer=Dense(output)(layer)
    self.model=keras.Model(inputs=inputLayer,outputs=outputLayer)
    self.model.compile(optimizer='Adam',loss='mse')

  def saveModel(self):
      self.model.save_weights(f"{self.modelName}")
      # print("model saved!")

  def loadModel(self):
      self.model.load_weights(f"{self.modelName}")
      # print("model sucsessfuly loaded!")

  def getAction(self,state):
    if np.random.random()<=self.exploreRate:
      return np.random.choice([i for i in range(env.action_space.n)])
    else:
      return np.argmax(self.model.predict(np.expand_dims(state,axis=0))[0])

  def exploreDecay(self):
      self.exploreRate=max(self.exploreRate*self.exploreRateDecay,self.minimumExploreRate)

  def learn(self,batchSize=64):
    if self.memory.size>batchSize:
      states,actions,rewards,nextStates,done=self.memory.sample(batchSize)
      qState=self.model.predict(states)
      qNextState=self.model.predict(nextStates)
      maxActions=np.argmax(qNextState,axis=1)
      batchIndex = np.arange(batchSize-1, dtype=np.int32)
      qState[batchIndex,actions]=(rewards+(self.gamma*qNextState[batchIndex,maxActions.astype(int)]*(1-done)))
      _=self.model.fit(x=states,y=qState,verbose=0)
      self.learnThreshold+=1
      self.exploreDecay()
      if(self.learnThreshold%self.copyNetsCycle)==0:
        self.tModel.set_weights(self.model.get_weights())
        self.saveModel("DoubleDQN_LunarLanderV2.h")
        self.learnThreshold=0

agent=Agent(stateShape=env.observation_space.shape[0],actionShape=env.action_space.n)
agent.loadModel()

agent.exploreRate=1
averageRewards=[]
totalRewards=[]
for i in range(1,500):
  done=False
  state=env.reset()
  rewards=0
  while not done:
    action=agent.getAction(state)
    nextState,reward,done,info=env.step(action)
    agent.memory.save(state,action,reward,nextState,int(done))
    rewards+=reward
    state=nextState
    agent.learn(batchSize=256)
  totalRewards.append(rewards)
  averageRewards.append(sum(totalRewards)/len(totalRewards))
  print(f"episode: {i}   reward: {rewards}  avg so far:{averageRewards[-1]} exploreRate:{agent.exploreRate}")

plt.title(f'Total Rewards')
plt.yscale('symlog')
plt.plot(totalRewards)
plt.savefig("Total Rewards",dpi=200)
plt.clf()
plt.title(f'Average Rewards')
plt.yscale('symlog')
plt.plot(averageRewards)
plt.savefig("Average Rewards",dpi=200)
