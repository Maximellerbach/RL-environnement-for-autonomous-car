import math
import random
from collections import deque

import cv2
import numpy as np
from keras.activations import relu
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, load_model, save_model
from keras.optimizers import Adam

import env

def log(score):

    val.append(score)

    valog_arr = np.array(val)
    np.save('log.npy',valog_arr)

inv = [2,1,0]

GAMMA = 0.99

MEMORY_SIZE = 10000
BATCH_SIZE = 128

EXPLORATION_MAX = 0.1
EXPLORATION_MIN = 0.05
EXPLORATION_DECAY = 0.1

val = []


class Solver():
    def __init__(self, observation_space, action_space):
        
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()

        self.model.add(Conv2D(16, (3,3), activation = "relu", padding = "same", input_shape=(observation_space)))
        self.model.add(MaxPooling2D())
        self.model.add(Dropout(0.1))

        self.model.add(Conv2D(8, (3,3), activation = "relu", padding = "same"))
        self.model.add(MaxPooling2D())
        self.model.add(Dropout(0.1))

        self.model.add(Conv2D(4, (3,3), activation = "relu", padding = "same"))
        self.model.add(MaxPooling2D())
        self.model.add(Dropout(0.1))

        self.model.add(Conv2D(2, (3,3), activation = "relu", padding = "same"))
        self.model.add(MaxPooling2D())
        self.model.add(Dropout(0.1))

        self.model.add(Flatten())
        self.model.add(Dense(32, use_bias=None, activation="relu"))
        self.model.add(Dense(16, use_bias=None, activation="relu"))

        self.model.add(Dense(self.action_space, use_bias=None, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam())

        self.model.summary()

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
            if done == False:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))

            q_values = self.model.predict(state)
            q_values[0][action] = q_update

            self.model.fit(state, q_values, verbose=0, batch_size=BATCH_SIZE)

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
    #dqn_solver.model = load_model('model\\dqn_solver2_12074.0.h5')
    
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
                
                # reset env
                car_obj.reset()
                dqn_solver.experience_replay()

                break
            

        if tot_reward>best:
            save_model(dqn_solver.model, 'model\\dqn_solver2_'+str(tot_reward)+'.h5')
            best = tot_reward
            print('woaw nice score!')

        log(tot_reward)


if __name__ == "__main__":
    autonomousCar()
