#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1a_problem import q1a_problem

def q1a_solver(problem: q1a_problem):
    astarData = astar_initialise(problem)
    num_expansions = 0
    terminate = False
    while not terminate:
        num_expansions += 1
        terminate, result = astar_loop_body(problem, astarData)
    print(f'Number of node expansions: {num_expansions}')
    return result

#-------------------#
# DO NOT MODIFY END #
#-------------------#


class AStarData:
    # YOUR CODE HERE
    def __init__(self):
        # frontier queue (using a priority queue from util)
        self.x = util.PriorityQueue()

        # explored set
        self.y = {}


def astar_initialise(problem: q1a_problem):
    astarData = AStarData()

    # get start and goal states
    start = problem.getStartState()
    goal = problem.getGoalState()

    # push the start state into the frontier queue with heuristic value
    astarData.x.push((start, []), astar_heuristic(start, goal))
    return astarData


def astar_loop_body(problem: q1a_problem, astarData: AStarData):
    # no solution found if frontier queue is empty
    if astarData.x.isEmpty():
        return True, None

    # get state with lowest f value from frontier queue
    current, path = astarData.x.pop()

    # goal is reached so return path
    if problem.isGoalState(current):
        return True, path

    # get new goal
    goal = problem.getGoalState()

    # loop through all successors of current state
    for successor, action, _ in problem.getSuccessors(current):

        # add the action
        newPath = path + [action]

        # if successor has not been explored
        if successor not in astarData.y:

            # calculate priority as path cost + heuristic value
            priority = len(newPath) + astar_heuristic(successor, goal)

            # add to frontier
            astarData.x.push((successor, newPath), priority)

            # set to explored
            astarData.y[successor] = current

    # continue exploring
    return False, None


def astar_heuristic(current, goal):
    # use manhattan distance as the heuristic
    return util.manhattanDistance(current, goal)
