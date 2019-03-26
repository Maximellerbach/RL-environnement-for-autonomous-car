# RL-environnement-for-autonomous-car
In this repo, I used some math and image manipulation skills to create my own reinforcement learning environnement for autonomous car 

![training](/docs/training.JPG)
![state](/docs/state.JPG)

this environnement is simple: a race track with white borders, 
in this environnement, a car (represented by a red point) is evolving, his goal is too survive as long as possible (so make as many laps as possible)

rewards are : 
* if the car is going forward then reward = current speed
* if the car is turning then reward = half the current speed (to prevent from turning too much)
* if the car is making an half turn reward = - 45
* if the car is going out of the track reward = - 150 and break the run/ cause respawn + AI training
