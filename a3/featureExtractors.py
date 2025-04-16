# This file contains feature extraction functions

import sys
import util
from pacman import Directions
from game import Directions, Actions
# from perceptronPacman import PerceptronPacman
import samples
import numpy as np
import math

# a bit quick and dirty but we use feature names in alphabetical order to make sure they are always the same order in the numpy feature vector.
# We use this here, in perceptronPacman and in the Q3 agent
FEATURE_NAMES =[ 
    'closestFood', 
    'closestFoodNow',
    'closestGhost',
    'closestGhostNow',
    'closestScaredGhost',
    'closestScaredGhostNow',
    'eatenByGhost',
    'eatsCapsule',
    'eatsFood',
    "foodCount",
    'foodWithinFiveSpaces',
    'foodWithinNineSpaces',
    'foodWithinThreeSpaces',  
    'furthestFood', 
    'numberAvailableActions',
    "ratioCapsuleDistance",
    "ratioFoodDistance",
    "ratioGhostDistance",
    "ratioScaredGhostDistance"
    ]

def walls(state):
    # Returns a list of (x, y) pairs of wall positions
    #
    # This version just returns all the current wall locations
    # extracted from the state data.  In later versions, this will be
    # restricted by distance, and include some uncertainty.

    wallList = []
    wallGrid = state.getWalls()
    width = wallGrid.width
    height = wallGrid.height
    for i in range(width):
        for j in range(height):
            if wallGrid[i][j] == True:
                wallList.append((i, j))
    return wallList


def inFront(object, facing, state):
    # Returns true if the object is along the corridor in the
    # direction of the parameter "facing" before a wall gets in the
    # way.

    pacman = state.getPacmanPosition()
    pacman_x = pacman[0]
    pacman_y = pacman[1]
    wallList = walls(state)

    # If Pacman is facing North
    if facing == Directions.NORTH:
        # Check if the object is anywhere due North of Pacman before a
        # wall intervenes.
        next = (pacman_x, pacman_y + 1)
        while not next in wallList:
            if next == object:
                return True
            else:
                next = (pacman_x, next[1] + 1)
        return False

    # If Pacman is facing South
    if facing == Directions.SOUTH:
        # Check if the object is anywhere due North of Pacman before a
        # wall intervenes.
        next = (pacman_x, pacman_y - 1)
        while not next in wallList:
            if next == object:
                return True
            else:
                next = (pacman_x, next[1] - 1)
        return False

    # If Pacman is facing East
    if facing == Directions.EAST:
        # Check if the object is anywhere due East of Pacman before a
        # wall intervenes.
        next = (pacman_x + 1, pacman_y)
        while not next in wallList:
            if next == object:
                return True
            else:
                next = (next[0] + 1, pacman_y)
        return False

    # If Pacman is facing West
    if facing == Directions.WEST:
        # Check if the object is anywhere due West of Pacman before a
        # wall intervenes.
        next = (pacman_x - 1, pacman_y)
        while not next in wallList:
            if next == object:
                return True
            else:
                next = (next[0] - 1, pacman_y)
        return False
    

"Feature extractors for Pacman game states"
class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features


def basicFeatureExtractorPacman(state):
    """
    A basic feature extraction function.
    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions
    ##
    """
    features = util.Counter()
    for action in state.getLegalActions():
        successor = state.generateSuccessor(0, action)
        foodCount = successor.getFood().count()
        featureCounter = util.Counter()
        featureCounter['foodCount'] = foodCount
        features[action] = featureCounter

    return features, state.getLegalActions()


def enhancedFeatureExtractorPacman(state):
    """
    Your feature extraction playground.
    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions
    ##
    """

    features = basicFeatureExtractorPacman(state)[0]
    for action in state.getLegalActions():
        features[action] = util.Counter(features[action], **furtherEnhancedPacmanFeatures(state, action))
    return features, state.getLegalActions()


def furtherEnhancedPacmanFeatures(state, action):
    """
    For each state, this function is called with each legal action.
    It should return a counter with { <feature name> : <feature value>, ... }
    """
    features = util.Counter()
    successor = state.generateSuccessor(0, action)  # Generate successor for current state
    # currentGhostPosition = [ghost.getPosition() for ghost in state.getGhostStates()]

    pacmanPosition = successor.getPacmanPosition()
    pacmanPositonCurrent = state.getPacmanPosition()
    foodList = successor.getFood().asList()  # Make list of food
    capsuleList = successor.getCapsules()
    ghostPosition = [ghost.getPosition() for ghost in successor.getGhostStates() if ghost.scaredTimer == 0]  # Find ghost positions
    scaredGhostPosition = [ghost.getPosition() for ghost in successor.getGhostStates() if ghost.scaredTimer > 0]
    possibleActions = successor.getLegalActions()

    max_dist = successor.data.layout.height * successor.data.layout.width

    # this only happens when we win the game
    if not foodList:
        # set to 0 because closer food is good
        closestFood = 0
        closestFoodNow = 1  # if we won the game then we just ate the last food and so must have been 1 away from it previously
        furthestFood = 0
        foodWithinThreeSpaces = 0
        foodWithinFiveSpaces = 0
        foodWithinNineSpaces = 0
    else:  # Find minimum distance to food
        foodDistances = np.array([util.manhattanDistance(i, pacmanPosition) for i in foodList])
        closestFoodNow = min([util.manhattanDistance(i, pacmanPositonCurrent) for i in foodList])
        
        # if we ate a food then set the distance to 0. Otherwise distance might increase which makes eating a food look bad.
        if state.getNumFood() - successor.getNumFood() == 1:
            closestFood = 0
        else:
            closestFood = min(foodDistances)
        furthestFood = max(foodDistances)
        foodWithinThreeSpaces = len(foodDistances[foodDistances <= 3])
        foodWithinFiveSpaces = len(foodDistances[foodDistances <= 5])
        foodWithinNineSpaces = len(foodDistances[foodDistances <= 9])

    if not ghostPosition:
        closestGhostNextState = max_dist
        closestGhostNow = max_dist
    else:
        closestGhostNextState = min(util.manhattanDistance(i, pacmanPosition) for i in ghostPosition)
        closestGhostNow = min(util.manhattanDistance(i, pacmanPositonCurrent) for i in ghostPosition)

    if not scaredGhostPosition:
        closestScaredGhost = max_dist
        closestScaredGhostNow = max_dist
    else:
        closestScaredGhost = min(util.manhattanDistance(i, pacmanPosition) for i in scaredGhostPosition)
        closestScaredGhostNow = min(util.manhattanDistance(i, pacmanPositonCurrent) for i in scaredGhostPosition)

    if not capsuleList:
        closestCapsule = 0
        closestCapsuleNow = 0 # set to 0 because closer capsule is good
    else:  # Find minimum distance to food
        closestCapsule = min([util.manhattanDistance(i, pacmanPosition) for i in capsuleList])
        closestCapsuleNow = min([util.manhattanDistance(i, pacmanPositonCurrent) for i in capsuleList])
    
    # weird things happen very rarely with the scared ghosts which makes this 0. This is never 0 because pacman is eaten
    # I don't know what causes it and it's so rare that for now it's just safest to set this to 1 if it happens.
    if closestGhostNow != 0:
        features["ratioGhostDistance"] = closestGhostNextState / closestGhostNow
    else:
        features["ratioGhostDistance"] = 1
    if closestScaredGhostNow != 0: 
        features["ratioScaredGhostDistance"] = closestScaredGhost / closestScaredGhostNow
    else:
        features["ratioScaredGhostDistance"] = 1
    
    # these features improve when you add the one both denominator and numerator
    features["ratioCapsuleDistance"] = (1 + closestCapsule) / (1 + closestCapsuleNow)
    features["ratioFoodDistance"] = (1 + closestFood) / (1 + closestFoodNow)

    # All features listed with reasoning
    features["foodCount"] = successor.getFood().count()  # We want to decrease food
    features['closestFood'] = 10 / (1+closestFood)  # Pacman wants to move towards food
    features['closestFoodNow'] = 10 / (1+closestFoodNow)
    features["furthestFood"] = furthestFood  # Pacman wants to move towards food 
    features['closestGhost'] = closestGhostNextState  # Pacman wants to stay away from ghosts
    features['closestGhostNow'] = closestGhostNow
    features["closestScaredGhost"] = closestScaredGhost  # Pacman wants to capture scared ghosts
    features['closestScaredGhostNow'] = closestScaredGhostNow
    features["closestCapsule"] = closestCapsule   # Pacman needs to eat capsule to capture scared ghosts
    features["closestCapsuleNow"] = closestCapsuleNow   # Pacman needs to eat capsule to capture scared ghosts
    features["eatsFood"] = (state.getNumFood() - successor.getNumFood()) + np.random.normal(loc=0, scale=0.25) # Pacman wants to eat food
    features["eatenByGhost"] = successor.isLose()
    features["eatsCapsule"] = len(state.getCapsules()) - len(successor.getCapsules())  # Pacman wants to eat capsules
    features["foodWithinThreeSpaces"] = foodWithinThreeSpaces  # Pacman wants to move towards areas with lots of food
    features["foodWithinFiveSpaces"] = foodWithinFiveSpaces # Pacman wants to move towards areas with lots of food
    features["foodWithinNineSpaces"] = foodWithinNineSpaces # Pacman wants to move towards areas with lots of food
    features["numberAvailableActions"] = len(possibleActions) # Pacman wants to move towards areas with lots of food
    
    return features
