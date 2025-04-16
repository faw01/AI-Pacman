import random

import util
from agents.ghostAgents import GhostAgent
from game import Actions, Agent, Directions
from util import manhattanDistance


class RandomGhost( GhostAgent ):
    "A ghost that chooses a legal action uniformly at random."
    def getDistribution( self, state ):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ): dist[a] = 1.0
        dist.normalize()
        return dist
