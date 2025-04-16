# PacmanValueIterationAgent.py
# -----------------------
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
import util

from agents.learningAgents import ValueEstimationAgent
from game import Grid, Actions, Directions
import math
from pacman import GameState
import random
import numpy as np
import json


class Q1Agent(ValueEstimationAgent):
    """
    Q1 agent is a ValueIterationAgent takes a Markov decision process
    (see pacmanMDP.py) on initialization and runs value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp="PacmanMDP", discount=0.5, iterations=200, pretrained_values=None, json_param_file=None, maze_size=None, save_values=False):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will likely use:
              self.MDP.getStates()
              self.MDP.getPossibleActions(state)
              self.MDP.getTransitionStatesAndProbs(state, action)
              self.MDP.getReward(state, action)
              self.MDP.isTerminal(state)
        """
        mdp_func = util.import_by_name('./', mdp)
        self.mdp_func = mdp_func

        print('[Q1Agent] using mdp ' + mdp_func.__name__)

        if json_param_file and maze_size:
            with open(json_param_file) as parameters_json:
                params = json.load(parameters_json)
                self.discount = params[maze_size]["gamma"]
        else:
            self.discount = float(discount)
        
        self.iterations = int(iterations)

        if pretrained_values:
            self.values = np.loadtxt(pretrained_values)
        else:
            self.values = None

        # do we want to save values after training? Will be string if read from command line so eval() the string to get a boolean
        if type(save_values) == str:
            self.save_values_after_training = eval(save_values)
        else:
            self.save_values_after_training = save_values

    def getValue(self, state: tuple):
        """
        Takes an (x,y) tuple and returns the value of the state (computed in registerInitialState or from pretrained values).
        """
        return self.values[state[0], state[1]]

    def getAction(self, gameState: GameState):
        """
        Returns the action to take at the a location according to the values

        Note: reaching any positive terminal state is considered winning the game and results in +500 points
        To achieve this we need the ReachedPositiveTerminalStateException
        """

        pacman_location = gameState.getPacmanPosition()
        if pacman_location in self.MDP.getFoodStates():
            raise util.ReachedPositiveTerminalStateException("Reached a Positive Terminal State")
        else:
            best_action = self.computePolicyFromValues(pacman_location)
            return self.MDP.applyNoiseToAction(pacman_location, best_action)

    def registerInitialState(self, gameState: GameState):

        # set up the mdp with the agent starting state
        self.MDP = self.mdp_func(gameState)

        # if we haven't solved the mdp yet or are not using pretrained weights
        if self.values is None:

            print("solving MDP")
            self.values = np.zeros((self.MDP.grid_width, self.MDP.grid_height))

            #-------------------#
            # DO NOT MODIFY END #
            #-------------------#
            
            # VALUE ITERATION STARTS HERE
            # util.raiseNotDefined()
            
            for _ in range(self.iterations):
                updated_values = np.zeros_like(self.values)
                for x in range(self.MDP.grid_width):
                    for y in range(self.MDP.grid_height):
                        current_state = (x, y)
                        if self.MDP.isTerminal(current_state):
                            updated_values[x, y] = self.MDP.getReward(current_state, None, None)
                        else:
                            legal_actions = self.MDP.getPossibleActions(current_state)
                            if legal_actions:
                                updated_values[x, y] = max(
                                    self.computeQValueFromValues(current_state, action)
                                    for action in legal_actions
                                )
                self.values = updated_values

            # VALUE ITERATION ENDS HERE
            # Save the learnt values to a file for you if want to inspect them
            if self.save_values_after_training:
                print("here")
                np.savetxt(f"./models/Q1/{gameState.data.layout.layoutFileName[:-4]}.model", self.values,
                        header=f"{{'discount': {self.discount}, 'iterations': {self.iterations}}}")

    def computeQValueFromValues(self, state: tuple, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()    
        q_value = 0.0
        for next_state, transition_prob in self.MDP.getTransitionStatesAndProbs(state, action):
            immediate_reward = self.MDP.getReward(state, action, next_state)
            future_value = self.discount * self.values[next_state[0]][next_state[1]]
            q_value += transition_prob * (immediate_reward + future_value)
        return q_value

    def computePolicyFromValues(self, state: tuple):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.
        """

        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        best_action = None
        best_q_value = float('-inf')

        for action in self.MDP.getPossibleActions(state):
            current_q_value = self.computeQValueFromValues(state, action)
            if current_q_value > best_q_value:
                best_q_value = current_q_value
                best_action = action

        return best_action