# searchAgents.py
# ---------------
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


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""
import glob
import logging
import time

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState, GameStateData


class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        GameStateData.verbose = False
        function = util.import_by_name('./solvers', fn)
        problem = util.import_by_name('./problems', prob)

        self.searchFunction = lambda x: function(x)
        self.searchType =  lambda x: problem(x)
        self.actionIndex: int = 0

        print('[SearchAgent] using search function ' + function.__name__)
        print('[SearchAgent] using problem type ' + problem.__name__)

    def registerInitialState(self, state: GameState):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem.
        self.actionIndex = 0  # Reset action index.
        self.actions  = self.searchFunction(problem) # Find a path.
        totalCost = len(self.actions)
        print('Path found with total cost of %d in %.10f seconds' % (totalCost, time.time() - starttime))

    def getAction(self, state: GameState):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            logger = logging.getLogger('root')
            logger.info('Out of actions')
            logger.info(f"Remaining Food: {state.getNumFood()}")
            raise util.OutOfActions('Out of actions')