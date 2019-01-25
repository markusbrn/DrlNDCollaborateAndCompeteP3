# 1. Introduction
The goal of this project is to develop and test a (Multi Agent Policy Gradient) Reinforcement Learning Algorithm that is able to successfully control two agents that play tennis and try to keep the ball in the air for as long as possible.
The agents are rewarded individually. If an agent is able to hit the ball over the net it receives a reward of +0.1; if it lets the ball hit the ground or get out of bounds it receives a reward of -0.01.
To control the agents, for each agent two continuous actions are available which correspond to horizontal motion and jumping. The state space consists of 8 variables for each robot (position and velocity of ball and racket).
The environment is considered solved when the agents achieve a reward of +0.5 over 100 consecutive episodes. The reward per episode is calculated by summing the rewards for each agent over one episode This gives two scores per episode. For the calculation of the averaged reward the bigger value is chosen.

# 2. Installation
To install the project, the project repository has to be cloned with the command "git clone https://github.com/markusbrn/DrlNDCollaborateAndCompeteP3". Next please set up your python environment correctly. You can find the instructions how to do so here (https://github.com/udacity/deep-reinforcement-learning#dependencies).
Finally you have to download the environment that is used to simulate the robot world. Please follow the links below (depending on the operating system you are using):

- Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
- Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip
- Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip
- Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip

The unzipped environment files have to be copied in the project repository folder.

# 3. Training and Testing the ML Agent
In order to train and test the agent please start and run the jupyter notebook file from the project repository.