#WRITTEN BY MOHAMMAD ASADOLAHI mohammad.e.asadolahi@gmail.com
#https://github.com/mohammadAsadolahi

import numpy as np
import gym
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
import keras
import numpy as np
from google.colab import drive
drive.mount('/Drive')

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
  def __init__(self,stateShape,actionShape,epsilon,gamma,saveAfterIterations=1000):
      self.gamma=gamma
      self.epsilon=epsilon
      self.actionShape=actionShape
      self.memory=replayBuffer(1000000,stateShape)
      self.buildModel(stateShape,actionShape)
      self.saveAfterIterations=saveAfterIterations
      self.updateIterations=0
  
  def buildModel(self,input,output):
    inputLayer=keras.Input(shape=(input,))
    layer=Dense(256,activation='relu')(inputLayer)
    layer=Dense(256,activation='relu')(layer)
    outputLayer=Dense(output)(layer)
    self.model=keras.Model(inputs=inputLayer,outputs=outputLayer)
    self.model.compile(optimizer='Adam',loss='mse')

  def saveModel(self,modelName):
      self.model.save_weights(f"/Drive/MyDrive/LunarLanderModelWeights/{modelName}")

  def loadModel(self,modelName):
      self.model.load_weights(f"/Drive/MyDrive/LunarLanderModelWeights/{modelName}")
      self.tModel.set_weights(self.model.get_weights())

  def getAction(self,state):
    q=self.model.predict(np.expand_dims(state,axis=0))[0]
    if np.random.random()<=self.epsilon:
      return np.random.choice([i for i in range(env.action_space.n)])
    else:
      return np.argmax(q)

  def learn(self,batchSize=64):
    if self.memory.size>batchSize:
      states,actions,rewards,nextStates,done=self.memory.sample(batchSize)
      qState=self.model.predict(states)
      qNextState=self.model.predict(nextStates)
      batchIndex = np.arange(batchSize-1, dtype=np.int32)
      qState[batchIndex,actions]+=(rewards+(self.gamma*np.max(qNextState,axis=1)))-qState[batchIndex,actions]
      _=self.model.fit(x=states,y=qState,verbose=0)
      self.updateIterations+=1
      if(self.saveAfterIterations<self.updateIterations)
        self.saveModel("DQN_LunarLanderV2.h")
        self.updateIterations=0

agent=Agent(stateShape=env.observation_space.shape[0],actionShape=env.action_space.n\
            ,epsilon=0,gamma=0.99)

totalRewards=[]
for i in range(500):
  done=False
  state=env.reset()
  rewards=0
  while not done:
    action=agent.getAction(state)
    nextState,reward,done,info=env.step(action)
    agent.memory.save(state,action,reward,nextState,int(done))
    rewards+=reward
    state=nextState
    agent.learn()
  totalRewards.append(rewards)
  print(f"episode: {i+1}   reward: {rewards}  avg so far:{np.mean(totalRewards[max(0, i-100):(i+1)])}")
