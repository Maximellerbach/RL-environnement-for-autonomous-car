import math
import random
from collections import deque

import numpy as np
from keras.activations import relu, tanh
from keras.layers import Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, load_model, save_model
from keras.optimizers import Adam

import env
import autolib

val = []

def log(score):

    val.append(score)

    valog_arr = np.array(val)
    np.save('log.npy',valog_arr)


GAMMA = 0.99

MEMORY_SIZE = 1000000
BATCH_SIZE = 128

EXPLORATION_MAX = 0.1
EXPLORATION_MIN = 0.02
EXPLORATION_DECAY = 0.9


class Solver():
    def __init__(self, observation_space, action_space):
        
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()

        self.model.add(Convolution2D(16, (3,3), activation = "relu", padding = "same", input_shape=(observation_space)))
        self.model.add(Dropout(0.1))
        self.model.add(MaxPooling2D())

        self.model.add(Convolution2D(16, (3,3), activation = "relu", padding = "same"))
        self.model.add(MaxPooling2D())
        
        self.model.add(Convolution2D(16, (3,3), activation = "relu", padding = "same"))
        self.model.add(MaxPooling2D())

        

        self.model.add(Flatten())
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(32, activation="relu"))

        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam())

        #self.model.summary()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values)

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)

        for state, action, reward, state_next, done in batch:
            q_update = reward

            q_values = self.model.predict(state)
            q_values[0][action] = q_update

            self.model.fit(state, q_values, verbose=0, batch_size=16)

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)



def autonomousCar():
    
    car_obj = env.car()
    car_obj.speed = 16 #simulation accuracy / speed
    car_obj.turnangle = math.pi*car_obj.speed/250 # turning angle

    observation_space = car_obj.get_view()[0].shape
    print(observation_space)
    action_space = 3
    dqn_solver = Solver(observation_space, action_space)
    #dqn_solver.model = load_model('model\\dqn_solver1_119738.h5')
    
    best = 4000
    run = 0
    while True:
        run += 1
        state = car_obj.get_view()
        tot_reward = 0

        while True:

            car_obj.render()

            action = dqn_solver.act(state)
            state_next, reward, done = car_obj.step(action)

            tot_reward+=reward 

            dqn_solver.remember(state, action, reward, state_next, done)
            state = state_next

            if done == True:
                print("Run: " + str(run) + ", score: " + str(tot_reward))
                
                car_obj.y, car_obj.x = car_obj.spawn
                car_obj.vector = np.array([0, 1])
                car_obj.angle = 0
                car_obj.visu = np.copy(car_obj.resized)

                dqn_solver.experience_replay()

                break
            

        if tot_reward>best:
            save_model(dqn_solver.model, 'model\\dqn_solver2_'+str(tot_reward)+'.h5')
            best = tot_reward
            print('woaw nice score!')

        log(tot_reward)


if __name__ == "__main__":
    autonomousCar()