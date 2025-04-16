# gridworld.py
# ------------
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


import sys
import mdp
import environment
import util
import optparse
from pacman import GameState, GameStateData
from game import Actions, Directions
import random

from pacman import TIME_PENALTY

class PacmanMDP(mdp.MarkovDecisionProcess):
    """
    An MDP version of Pac-Man
    """
    def __str__(self):
        return str(self.__class__.__module__)

    def __init__(self, startingGameState: GameState):

        # parameters
        self.livingReward = 0.0
        self.noise = 0.2

        # pacman start state
        self.startingPosition = startingGameState.getPacmanPosition()

        # get the positive terminals by finding the food
        self.walls = startingGameState.getWalls()
        self.grid_width = self.walls.width
        self.grid_height = self.walls.height


        self.all_states = set([(x,y) for x in range(self.grid_width) for y in range(self.grid_height) if not self.walls[x][y]])
        
        # get the food and ghost exit states
        self.food_states = set([state for state in self.all_states if startingGameState.hasFood(state[0],state[1])])
        self.ghost_states = set(startingGameState.getGhostPositions())

        # We need a true terminal state where all of the "food exit" and "ghost exit" states can exit to. 
        # We set this state to be (0,0) which is always a wall in the map.
        self.true_terminal_state = (0, 0)
        self.all_states.add(self.true_terminal_state)


    def setNoise(self, noise):
        """
        The probability of moving in an unintended direction.
        """
        self.noise = noise


    def getPossibleActions(self, state):
        """
        Returns list of valid actions for 'state'.

        Note that you CANNOT request moves into walls.
        the food and ghost states have the "EXIT" action associated with it, so we can still call 
        methods like getTransitionStatesAndProbs which requires a state and action. 
        The terminal state has an action to STAY forever which gives 0 reward
        """

        if self.isTerminal(state):
            return ['STAY']
        elif state in self.getFoodStates() or state in self.getGhostStates():
            return ['EXIT']

        possible_actions = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            next_state = Actions.getSuccessor(state, action)
            next_x, next_y = int(next_state[0]), int(next_state[1])
            if self.isAllowed(next_x, next_y):
                possible_actions.append(action)

        return possible_actions

    def getStates(self):
        """
        Return list of all states.
        """
        return self.all_states
    
    def getFoodStates(self):
        return self.food_states
    
    def getGhostStates(self):
        return self.ghost_states

    def getReward(self, state, action, nextState):
        """
        Get reward for state, action transition.

        Note that the reward depends only on the state being departed as in the value iteration exmample in the lectures. 
        We don't actually need the action or nextState arguments but they're included to be consistent with the standard definition of reward function
        All non food or ghost states have a reward of -1 for any action
        Food states have a reward of 510 for exiting, with 10 coming from consumong a food and 500 for winning the game.
        Ghost states have a reward of -500 for exiting
        The terminal state has a reward of 0. 
        """
        if self.isTerminal(state):
            return 0
        elif state in self.getFoodStates():
            return 510
        elif state in self.getGhostStates():
            return -500
        else:
            return -1*TIME_PENALTY

    def getStartState(self):
        return self.startingPosition

    def isTerminal(self, state):
        """
        Only the self.true_terminal_state state is *actually* a terminal state.
        The other "exit" states are technically non-terminals with
        a single action "exit" which leads to the true terminal state.
        This convention is to make the grids line up with the examples
        in the lecture content.
        """
        return state == self.true_terminal_state


    def getTransitionStatesAndProbs(self, state, action):
        """
        Returns list of (nextState, prob) pairs representing the states reachable from 'state' by taking 'action' along
        with their transition probabilities. 
        If an action would lead Pac-Man to hit a wall then the result is that he stays in place, and nextState=state.
        """

        if action not in self.getPossibleActions(state):
            raise "Illegal action!"

        # remain in the terminal state forever
        elif self.isTerminal(state):
            return [(self.true_terminal_state, 1)]
        # only place to go from an exit state is the terminal state
        elif state in self.getFoodStates() or state in self.getGhostStates():
            return [(self.true_terminal_state, 1)]

        # store the transitions
        successors = []
        cur_x, cur_y = state

        # check the next state is available. If yes then go to the state with probability 1 - self.noise
        forward = Actions.getSuccessor(state, action)
        forward_x, forward_y = int(forward[0]), int(forward[1])
        if self.isAllowed(forward_x, forward_y):
            successors.append(((forward_x, forward_y), 1-self.noise))
        else:
            # can't move forward so you'll stay in place
            successors.append(((cur_x, cur_y), 1-self.noise))

        # check if right is available. If yes then go with self.noise/2
        right = Actions.getSuccessor(state, Directions.RIGHT[action])
        right_x, right_y = int(right[0]), int(right[1])
        if self.isAllowed(right_x, right_y):
            successors.append(((right_x, right_y), self.noise/2))
        else:
            # can't move right so you'll stay in place
            successors.append(((cur_x, cur_y), self.noise/2))

        # check if left is available. If yes then self.noise/2
        left = Actions.getSuccessor(state, Directions.LEFT[action])
        left_x, left_y = int(left[0]), int(left[1])
        if self.isAllowed(left_x, left_y):
            successors.append(((left_x, left_y), self.noise / 2))
        else:
            # can't move left so you'll stay in place
            successors.append(((cur_x, cur_y), self.noise/2))

        successors = self.__aggregate(successors)
        return successors


    def applyNoiseToAction(self, state, action):
        """
        Used by the Q1 agent to simulate having the MDP chance of moving the wrong direction.
        """
        # print(action)
        sample = random.random()

        # try and move right
        if sample < self.noise / 2:
            # check if right is available. If yes then go with self.noise/2
            right = Actions.getSuccessor(state, Directions.RIGHT[action])
            right_x, right_y = int(right[0]), int(right[1])
            if self.isAllowed(right_x, right_y):
                return Directions.RIGHT[action]
            else:
                # can't move right so you'll stay in place
                return Directions.STOP

        elif sample < self.noise:
            # check if left is available. If yes then self.noise/2
            left = Actions.getSuccessor(state, Directions.LEFT[action])
            left_x, left_y = int(left[0]), int(left[1])
            if self.isAllowed(left_x, left_y):
                return Directions.LEFT[action]
            else:
                # can't move left so you'll stay in place
                return Directions.STOP

        else:
            # check if we can execute planned action. We need an if because if V values haven't converged then agent might think the best action is to walk into a wall.
            forward = Actions.getSuccessor(state, action)
            forward_x, forward_y = int(forward[0]), int(forward[1])
            if self.isAllowed(forward_x, forward_y):
                return action
            else:
                # can't move left so you'll stay in place
                return Directions.STOP

    def __aggregate(self, statesAndProbs):
        counter = util.Counter()
        for state, prob in statesAndProbs:
            counter[state] += prob
        newStatesAndProbs = []
        for state, prob in counter.items():
            newStatesAndProbs.append((state, prob))
        return newStatesAndProbs

    def isAllowed(self, x, y):
        if y < 0 or y >= self.grid_height: return False
        if x < 0 or x >= self.grid_width: return False
        return not self.walls[x][y]
