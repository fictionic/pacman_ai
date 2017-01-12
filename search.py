# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

class PathNode:
    def __init__(self, node, action=None, parent=None, cost=0):
        self.node = node
        self.action = action
        self.parent = parent
        self.cost = cost

def depthFirstSearch(problem):
    """
    Your DFS implementation goes here

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    # Check if the start state is the goal state, if so return an empty list
    if problem.isGoalState(problem.getStartState):
        return []

    frontier = util.Stack()
    frontier.push(PathNode(problem.getStartState()))
    visited = set()
    while not frontier.isEmpty():
        curNode = frontier.pop()
        visited.add(curNode.node)
        for (successor, action, _) in problem.getSuccessors(curNode.node):
            if successor not in visited:
                childNode = PathNode(successor, action, curNode)
                # Check if nodes are goal states before adding them to the frontier
                if problem.isGoalState(childNode.node):
                    ret = [childNode.action]
                    while childNode.parent:
                        if childNode.parent.action:
                            ret.append(childNode.parent.action)
                        childNode = childNode.parent
                    ret.reverse()
                    return ret
                frontier.push(childNode)


def breadthFirstSearch(problem):
    """Your BFS implementation goes here. Like for DFS, your 
    search algorithm needs to return a list of actions that 
    reaches the goal.
    """
    # Check if the start state is the goal state, if so return an empty list
    if problem.isGoalState(problem.getStartState()):
        return []

    frontier = util.Queue()
    frontier.push(PathNode(problem.getStartState()))
    visited = set()
    while not frontier.isEmpty():
        curNode = frontier.pop()
        visited.add(curNode.node)
        for (successor, action, _) in problem.getSuccessors(curNode.node):
            if successor not in visited:
                childNode = PathNode(successor, action, curNode)
                # Check if nodes are goal states before adding them to the frontier
                if problem.isGoalState(childNode.node):
                    ret = [childNode.action]
                    while childNode.parent:
                        if childNode.parent.action:
                            ret.append(childNode.parent.action)
                        childNode = childNode.parent
                    ret.reverse()
                    return ret
                frontier.push(childNode)

def uniformCostSearch(problem):
    """Your UCS implementation goes here. Like for DFS, your 
    search algorithm needs to return a list of actions that 
    reaches the goal.
    """
    # Check if the start state is the goal state, if so return an empty list
    if problem.isGoalState(problem.getStartState()):
        return []

    frontier = util.PriorityQueue()
    frontier.push(PathNode(problem.getStartState()), 0)
    visited = set()
    while not frontier.isEmpty():
        curNode = frontier.pop()
        # Check if nodes are goal states when we pop them off the frontier
        if problem.isGoalState(curNode.node):
            ret = [curNode.action]
            while curNode.parent:
                if curNode.parent.action:
                    ret.append(curNode.parent.action)
                curNode = curNode.parent
            ret.reverse()
            return ret
        visited.add(curNode.node)
        for (successor, action, _) in problem.getSuccessors(curNode.node):
            if successor not in visited:
                newCost = problem.getCostOfActions([action])
                frontier.push(PathNode(successor, action=action, parent=curNode, cost=curNode.cost+newCost), newCost)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Your A* implementation goes here. Like for DFS, your 
    search algorithm needs to return a list of actions that 
    reaches the goal. heueristic is a heuristic function - 
    you can see an example of the arguments and return type
    in "nullHeuristic", above.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
