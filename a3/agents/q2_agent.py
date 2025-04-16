# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#
from game import *
from agents.learningAgents import ReinforcementAgent
from pacman import GameState

import random,util,math
import numpy as np
from game import Directions
import json


class Q2Agent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - epsilonGreedyActionSelection
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, pretrained_values=None, json_param_file=None, maze_size=None, save_values=False, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p Q2Agent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """

        self.index = 0  # This is always Pacman

        ReinforcementAgent.__init__(self, **args)

        # do we want to save values after training? Will be string if read from command line so eval() the string to get a boolean
        if type(save_values) == str:
            self.save_values_after_training = eval(save_values)
        else:
            self.save_values_after_training = save_values

        if json_param_file and maze_size:
            with open(json_param_file) as parameters_json:
                params = json.load(parameters_json)
                self.alpha = params[maze_size]["alpha"]
                self.epsilon = params[maze_size]["epsilon"]
                self.discount = params[maze_size]["gamma"]

        if pretrained_values:
            flattenedQ = np.loadtxt(pretrained_values)
            width, height = flattenedQ.shape
            self.Q_values = flattenedQ.reshape(int(width/4), height, 4) # We only want 4 actions because STOP isn't allowed
            self.learningQvalues = False
            self.numTraining = 0 # no training
            self.epsilon = 0.0  # no exploration
            self.alpha = 0.0  # no learning
        else:
            self.Q_values = None
            self.learningQvalues = True
            self.epsilon_to_write = self.epsilon
            self.alpha_to_write = self.epsilon

    def registerInitialState(self, state: GameState):
        """
        Don't modify this method!
        """
        # We start out by finding the food locations so we can end an episode when we reach any of them
        walls = state.getWalls()
        grid_width, grid_height = walls.width, walls.height
        all_reachable_states = set([(x,y) for x in range(grid_width) for y in range(grid_height) if not walls[x][y]])
        self.food_locations = set([location for location in all_reachable_states if state.hasFood(location[0],location[1])])

        if self.Q_values is None:
            self.Q_values = np.zeros((state.getWalls().width, state.getWalls().height, 4)) # We only want 4 actions because stop isn't allowed
            self.learningQvalues = True

        elif self.isInTesting() and self.learningQvalues:
            self.learningQvalues = False

            if self.save_values_after_training:
                width, height, depth = self.Q_values.shape
                flattenedQ = self.Q_values.reshape((width*depth, height))
                np.savetxt(f"./models/Q2/{state.data.layout.layoutFileName[:-4]}.model", flattenedQ,
                        header=f"{{'gamma':{self.discount}, 'num_training':{self.numTraining}, 'epsilon':{self.epsilon_to_write}, 'alpha':{self.alpha_to_write}}}")

        # set epsilon to 0 for testing
        if self.isInTesting():
            print(self.epsilon, self.alpha)

        self.startEpisode()
        if self.episodesSoFar == 0:
            print('Beginning %d episodes of Training' % (self.numTraining))

    def getActionIndex(self, action):
        """
        This function maps an action to the correct index in the Q-table
        """
        if action == Directions.NORTH:
            return 0
        elif action == Directions.SOUTH:
            return 1
        elif action == Directions.EAST:
            return 2
        elif action == Directions.WEST:
            return 3

    def getPolicy(self, state: GameState):
        return self.computeActionFromQValues(state)

    def getValue(self, state: GameState):
        return self.computeValueFromQValues(state)
    
    def getLegalActions(self, game_state: GameState):
        """
        Gets the  the method from the base clase because 
        """
        actions = super().getLegalActions(game_state)
        if len(actions) > 0: actions.remove("Stop")
        return actions
    
    def getAction(self, state: GameState):
        """
        Uses epsilon greedy to select an action based on the agents Q table.
        """
        pacman_location = state.getPacmanPosition()
        if pacman_location in self.food_locations:
            raise util.ReachedPositiveTerminalStateException("Reached a Positive Terminal State")

        action = self.epsilonGreedyActionSelection(state)

        self.doAction(state, action)
        return action

    #-------------------#
    # DO NOT MODIFY END #
    #-------------------#

    def getQValue(self, state: tuple, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        x, y = state.getPacmanPosition()
        action_index = self.getActionIndex(action)
        return self.Q_values[x, y, action_index]

    def computeValueFromQValues(self, state: GameState):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.

          Note that if there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.

          HINT: You might want to use self.getLegalActions(state)
        """

        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        legal_actions = self.getLegalActions(state)
        if not legal_actions:
            return 0.0
        return max(self.getQValue(state, action) for action in legal_actions)


    def computeActionFromQValues(self, state: GameState):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
          HINT: This function should be a strict max over the Q values, not an epsilon greedy
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        legal_actions = self.getLegalActions(state)
        if not legal_actions:
            return None
        
        best_value = self.computeValueFromQValues(state)
        best_actions = [action for action in legal_actions 
                        if self.getQValue(state, action) == best_value]
        return random.choice(best_actions)
       

    def epsilonGreedyActionSelection(self, state: GameState):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.

        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
        HINT: You might want to use self.getLegalActions(state)
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        legal_actions = self.getLegalActions(state)
        if not legal_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        else:
            return self.computeActionFromQValues(state)
        

    def update(self, state: GameState, action, nextState: GameState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here using the Q value update equation

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        x, y = state.getPacmanPosition()
        action_index = self.getActionIndex(action)
        
        current_q = self.getQValue(state, action)
        next_max_q = self.computeValueFromQValues(nextState)
        
        updated_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.discount * next_max_q)
        
        self.Q_values[x, y, action_index] = updated_q

