import logging
import random

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState
from util import manhattanDistance


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

def scoreEvaluationFunctionImproved(currentGameState):

    # weights for heuristics
    LOSE_PENALTY = -100000000000
    WIN_BONUS = 100000000000
    FOOD_WEIGHT = 1
    GHOST_WEIGHT = 1

    # get pacman position, food, food grid and capsules
    start = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodGrid = food.asList()

    # check if pacman has lost or won 
    if currentGameState.isLose():
        return LOSE_PENALTY + scoreEvaluationFunction(currentGameState)
    if currentGameState.isWin():
        return WIN_BONUS + scoreEvaluationFunction(currentGameState)

    # check if all food is eaten
    if len(foodGrid) <= 0:
        findFood = 0
    else:
        # if not then find the nearest food
        findFood = radar(start, foodGrid, currentGameState)

        if findFood == float('inf'):
            findFood = min([util.manhattanDistance(start, food) for food in foodGrid])

        findFood = 1 / (findFood + 1)

    ghostDist = findMinimumManhattanGhostDistance(start, currentGameState.getGhostStates())
    # ghostDist = findNearestGhostDistance(start, currentGameState)
    ghostPenalty = ghostSensitivity(ghostDist)
    # ghostPenalty = 0
    
    return (GHOST_WEIGHT * ghostPenalty) + (FOOD_WEIGHT * findFood) + scoreEvaluationFunction(currentGameState)

def inverseH(value, weight, default=100000000000):
    return weight * value**-1 if value != 0 else default

def findNearestGhostDistance(pacman, gameState):
    ghostPosition = [ghost.getPosition() for ghost in gameState.getGhostStates()]
    return radar(pacman, ghostPosition, gameState)

def findMinimumManhattanGhostDistance(pacman, ghost_states):
    return min(util.manhattanDistance(pacman, ghost.getPosition()) for ghost in ghost_states)

def ghostSensitivity(distance):
    GHOST_PROXIMITY_PENALTY = -200
    return GHOST_PROXIMITY_PENALTY if distance < 2 else 0

def radar(position, goals, gameState):
    if not goals:
        return 0

    walls = gameState.getWalls()
    queue = util.Queue()
    queue.push((position, 0))
    explored = set([position])

    while not queue.isEmpty():
        position, distance = queue.pop()

        if position in goals:
            return distance

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = position
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            next_position = (nextx, nexty)

            if not walls[nextx][nexty] and next_position not in explored:
                queue.push((next_position, distance + 1))
                explored.add(next_position)

    return float('inf')

class Q2_Agent(Agent):

    def __init__(self, evalFn = 'scoreEvaluationFunctionImproved', depth = '3'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    @log_function
    def getAction(self, gameState: GameState):
        """
            Returns the minimax action from the current gameState using self.depth
            and self.evaluationFunction.

            Here are some method calls that might be useful when implementing minimax.

            gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

            gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

            gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        logger = logging.getLogger('root')
        logger.info('MinimaxAgent')
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        def alpha_beta_search(state):
            value, move = max_value(state, float('-inf'), float('inf'), 0)
            return move

        def max_value(state, alpha, beta, depth):
            if is_terminal(state, depth):
                return self.evaluationFunction(state), None

            v, move = float('-inf'), None
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                v2, a2 = min_value(successor, alpha, beta, depth, 1)
                if v2 > v:
                    v, move = v2, action
                alpha = max(alpha, v)
                if v >= beta:
                    return v, move
            return v, move

        def min_value(state, alpha, beta, depth, agent_index):
            if is_terminal(state, depth):
                return self.evaluationFunction(state), None

            v, move = float('inf'), None
            next_agent = agent_index + 1
            if next_agent == state.getNumAgents():
                next_agent = 0
                depth += 1

            for action in state.getLegalActions(agent_index):
                successor = state.generateSuccessor(agent_index, action)
                if next_agent == 0:
                    v2, a2 = max_value(successor, alpha, beta, depth)
                else:
                    v2, a2 = min_value(successor, alpha, beta, depth, next_agent)
                if v2 < v:
                    v, move = v2, action
                beta = min(beta, v)
                if v <= alpha:
                    return v, move
            return v, move

        def is_terminal(state, depth):
            return depth == self.depth or state.isWin() or state.isLose()

        return alpha_beta_search(gameState)