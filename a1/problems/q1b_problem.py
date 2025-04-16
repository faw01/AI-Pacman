import logging
import time
from typing import Tuple

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState


class q1b_problem:
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __str__(self):
        return str(self.__class__.__module__)

    def __init__(self, gameState: GameState):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.startingGameState: GameState = gameState

    @log_function
    def getStartState(self):
        # get pacman starting position
        start = self.startingGameState.getPacmanPosition()

        # return start state with food not eaten set to true
        return (start[0], start[1], True)

    @log_function
    def isGoalState(self, state):
        # unpack state
        x, y, foodNotEaten = state

        # check if current position is a food dot and food has been eaten
        return (x, y) in self.startingGameState.getFood().asList() and not foodNotEaten

    @log_function
    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """
        successors = []
        x, y, foodNotEaten = state
        # loop through all possible actions
        for action in [
            Directions.NORTH,
            Directions.SOUTH,
            Directions.EAST,
            Directions.WEST,
        ]:

            # get direction vector
            dx, dy = Actions.directionToVector(action)

            # calculate next position
            nextX, nextY = int(x + dx), int(y + dy)

            # check if next position is anything but a wall
            if not self.startingGameState.hasWall(nextX, nextY):
                # check if moving to this position eats the food
                nextFood = foodNotEaten and not self.startingGameState.hasFood(
                    nextX, nextY
                )

                # get next state
                nextState = (nextX, nextY, nextFood)

                # add the successor to the list with a cost of 1
                successors.append((nextState, action, 1))
        return successors

    def getGoalState(self):
        # get the position of all food dots
        return self.startingGameState.getFood().asList()
