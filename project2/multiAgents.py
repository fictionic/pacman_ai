# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
import sys

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
    """


    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        This evaluation function is not particularly good; using information
        from the game state would allow it to be much better, although still
        not as good as an agent that plans. You may find the information listed
        below helpful in later parts of the project (e.g., when designing
        an evaluation function for your planning agent).
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

def debug(s):
    # print(s)
    pass

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 1)
    """

    def getAction(self, gameState):
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
        # we're pacman
        return self.maxValueAction(0, 0, gameState)[1]

    def minValueAction(self, ghostIndex, curDepth, state):
        # if we're at a terminal state
        if state.isLose() or state.isWin():
            return self.evaluationFunction(state), None
        nextDepth = None
        if ghostIndex == state.getNumAgents() - 1:
            func = self.maxValueAction
            nextIndex = 0
            nextDepth = curDepth + 1
        else:
            func = self.minValueAction
            nextIndex = ghostIndex + 1
            nextDepth = curDepth
        v = None
        for action in state.getLegalActions(ghostIndex):
            child = state.generateSuccessor(ghostIndex, action)
            childV = func(nextIndex, nextDepth, child)[0]
            if v is None or childV < v:
                v = childV
        # if we're at a leaf
        if len(state.getLegalActions()) == 0:
            v = self.evaluationFunction(state)
        return v, None
        
    def maxValueAction(self, pacmanIndex, curDepth, state):
        v = None
        a = None
        # if we've reached the max depth, or if we're at a terminal state
        if curDepth == self.depth or state.isLose() or state.isWin():
            return self.evaluationFunction(state), None
        nextIndex = 1
        nextDepth = curDepth
        for action in state.getLegalActions(pacmanIndex):
            child = state.generateSuccessor(pacmanIndex, action)
            childV = self.minValueAction(nextIndex, nextDepth, child)[0]
            if v is None or childV > v:
                v = childV
                a = action
        # if we're at a leaf
        if len(state.getLegalActions()) == 0:
            v = self.evaluationFunction(state)
        return v, a
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        # we're pacman
        return self.maxValueAction(0, 0, gameState, -sys.maxint, sys.maxint)[1]

    def maxValueAction(self, pacmanIndex, curDepth, state, alpha, beta):
        # if we've reached the max depth, or if we're at a terminal state
        if curDepth == self.depth or state.isLose() or state.isWin():
            return self.evaluationFunction(state), None
        v = None
        a = None
        nextIndex = 1
        nextDepth = curDepth
        for action in state.getLegalActions(pacmanIndex):
            child = state.generateSuccessor(pacmanIndex, action)
            childV = self.minValueAction(nextIndex, nextDepth, child, alpha, beta)[0]
            if v is None or childV > v:
                v = childV
                a = action
            # prune of v>beta
            if beta is None or v > beta:
                return v, action
            alpha = max(v, alpha) if alpha is not None else v
        # if we're at a leaf
        if len(state.getLegalActions()) == 0:
            v = self.evaluationFunction(state)
        return v, a

    def minValueAction(self, ghostIndex, curDepth, state, alpha, beta):
        # if we're at a terminal state
        if state.isLose() or state.isWin():
            return self.evaluationFunction(state), None
        nextDepth = None
        if ghostIndex == state.getNumAgents() - 1:
            func = self.maxValueAction
            nextIndex = 0
            nextDepth = curDepth + 1
        else:
            func = self.minValueAction
            nextIndex = ghostIndex + 1
            nextDepth = curDepth
        v = None
        for action in state.getLegalActions(ghostIndex):
            child = state.generateSuccessor(ghostIndex, action)
            childV = func(nextIndex, nextDepth, child, alpha, beta)[0]
            if v is None or childV < v:
                v = childV
            # if the parent is pacman
            if alpha is None or v < alpha:
                return v, action
            beta = min(v, beta) if beta is not None else v
        # if we're at a leaf
        if len(state.getLegalActions()) == 0:
            v = self.evaluationFunction(state)
        return v, None


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.maxValueAction(0, 0, gameState)[1]
        
    def maxValueAction(self, pacmanIndex, curDepth, state):
        # if we've reached the max depth, or if we're at a terminal state
        if curDepth == self.depth or state.isLose() or state.isWin():
            return self.evaluationFunction(state), None
        v = None
        a = None
        nextIndex = 1
        nextDepth = curDepth
        for action in state.getLegalActions(pacmanIndex):
            child = state.generateSuccessor(pacmanIndex, action)
            childV = self.chanceValueAction(nextIndex, nextDepth, child)[0]
            if v is None or childV > v:
                v = childV
                a = action
        # if we're at a leaf
        if len(state.getLegalActions()) == 0:
            v = self.evaluationFunction(state)
        return v, a
        
    def chanceValueAction(self, ghostIndex, curDepth, state):
        # if we've reached the max depth, or if we're at a terminal state
        if state.isLose() or state.isWin():
            return self.evaluationFunction(state), None
        nextDepth = None
        if ghostIndex == state.getNumAgents() - 1:
            func = self.maxValueAction
            nextIndex = 0
            nextDepth = curDepth + 1
        else:
            func = self.chanceValueAction
            nextIndex = ghostIndex + 1
            nextDepth = curDepth
        numLegalActions = len(state.getLegalActions(ghostIndex))
        expectedV = 0.0
        for action in state.getLegalActions(ghostIndex):
            child = state.generateSuccessor(ghostIndex, action)
            childV = func(nextIndex, nextDepth, child)[0]
            expectedV += childV
        # if we're at a leaf
        if len(state.getLegalActions()) == 0:
            expectedV = self.evaluationFunction(state)
        expectedV /= numLegalActions
        return expectedV, None

#### helper things for evaluationFunction

def getNearestGhost(gameState):
    curPos = gameState.getPacmanPosition()
    ghostDist = None
    closestIndex = None
    for ghostIndex in xrange(gameState.getNumAgents()):
        if ghostIndex == 0:
            continue
        ghostPos = gameState.getGhostPosition(ghostIndex)
        dist = manhattanDistance(ghostPos, curPos)
        if ghostDist is None or dist < ghostDist:
            ghostDist = dist
            closestIndex = ghostIndex
    return closestIndex

def getNearestAngryGhost(gameState):
    curPos = gameState.getPacmanPosition()
    ghostDist = None
    closestIndex = None
    for ghostIndex in xrange(gameState.getNumAgents()):
        if ghostIndex == 0:
            continue
        ghostState = gameState.getGhostState(ghostIndex)
        # skip all of the scared ghosts for this
        if ghostState.scaredTimer > 0:
            continue
        ghostPos = gameState.getGhostPosition(ghostIndex)
        dist = manhattanDistance(ghostPos, curPos)
        if ghostDist is None or dist < ghostDist:
            ghostDist = dist
            closestIndex = ghostIndex
    return closestIndex

class Feature:
    def __init__(self, weight, valueFn):
        self.weight = weight
        self.valueFn = valueFn

features = []
# number of foods left
weight = 50
def fn(gameState):
    numFood = gameState.getNumFood()
    if numFood == 0:
        return 500
    return 1.0/numFood**2
feature = Feature(weight, fn)
features.append(feature)

# manhattan distance to nearest capsule 
weight = 0
def fn(gameState):
    ret = None
    curPos = gameState.getPacmanPosition()
    for capsule in gameState.getCapsules():
        dist = manhattanDistance(capsule, curPos)
        if ret is None or dist < ret:
            ret = dist
    if ret is None:
        return 0
    return 1.0/ret
feature = Feature(weight, fn)
features.append(feature)
# if nearest ghost is scared: distance to scared ghost;
# else negative distance to nearest ghost
weight = 20
def fn(gameState):
    ghostDist = None
    ghostIndex = getNearestGhost(gameState)
    if ghostIndex is None:
        return 0
    ghostState = gameState.getGhostState(ghostIndex)
    ghost = gameState.getGhostPosition(ghostIndex)
    curPos = gameState.getPacmanPosition()
    dist = manhattanDistance(ghost, curPos)
    if dist == 0:
        return -200
    ret = 1.0/(dist**2)
    if ghostState.scaredTimer > 0:
        return ret
    else:
        return -ret
feature = Feature(weight, fn)
features.append(feature)

# weight of food based on nearest not scared ghost
weight = 0
def fn(gameState):
    numFood = gameState.getNumFood()
    if numFood == 0:
        return 500
    nearestAngryGhost = getNearestAngryGhost(gameState)
    if nearestAngryGhost is None:
        return 0
    return nearestAngryGhost/gameState.getNumFood()
feature = Feature(weight, fn)
features.append(feature)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    ret = 0
    for feature in features:
        ret += feature.weight * feature.valueFn(currentGameState)
    return ret

# Abbreviation
better = betterEvaluationFunction

