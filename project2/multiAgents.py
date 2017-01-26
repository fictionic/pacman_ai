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
        if self.index == 0:
            # we're pacman
            return self.maxValueAction(0, gameState)[1]
        else:
            # we're a ghost (shouldn't be called)
            return self.minValueValue(self.index, gameState)[1]

    def minValueAction(self, searchDepth, state):
        func = self.maxValueAction if searchDepth - 1 % state.getNumAgents() == 0 else self.minValueAction
        v = None
        a = None
        # if we've reached the max depth
        if searchDepth == self.depth * state.getNumAgents():
            return self.evaluationFunction(state), None
        if state.isLose() or state.isWin():
            return state.getScore(), None
        vs = []
        debug('\t' * searchDepth + "min")
        debug('\t' * searchDepth + str(state.getLegalActions(self.index)))

        ghostIndex = searchDepth % state.getNumAgents()

        for action in state.getLegalActions(ghostIndex):
            child = state.generateSuccessor(ghostIndex, action)
            childV = func(searchDepth+1, child)[0]
            vs.append(childV)
            debug('\t' * searchDepth + "child: " + str(action) + " " + str(childV))
            if not v or childV < v:
                v = childV
                a = action
        debug('\t' * searchDepth + 'depth: ' + str(searchDepth))
        debug('\t' * searchDepth +'vs: ' + str(vs))
        debug('\t' * searchDepth +'v: ' + str(v))
        # if we're at a leaf
        if len(state.getLegalActions()) == 0:
            v = self.evaluationFunction(state)
        return v, a
        
    def maxValueAction(self, searchDepth, state):
        v = None
        a = None
        vs = []
        # if we've reached the max depth
        if searchDepth == self.depth * state.getNumAgents():
            return self.evaluationFunction(state), None
        if state.isLose() or state.isWin():
            return state.getScore(), None
        debug('\t' * searchDepth + "max")
        debug('\t' * searchDepth + str(state.getLegalActions(self.index)))
        for action in state.getLegalActions(self.index):
            child = state.generateSuccessor(self.index, action)
            childV = self.minValueAction(searchDepth+1, child)[0]
            if not v or childV > v:
                v = childV
                a = action
            vs.append(childV)
        debug('\t' * searchDepth +'depth: ' + str(searchDepth))
        debug('\t' * searchDepth +'vs: ' + str(vs))
        debug('\t' * searchDepth +'v: ' + str(v))
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

        if self.index == 0:
            # we're pacman
            return self.maxValueAction(0, gameState, -sys.maxint, sys.maxint)[1]
        else:
            # we're a ghost (shouldn't be called)
            return self.minValueValue(self.index, gameState, -sys.maxint, sys.maxint)[1]

    def maxValueAction(self, searchDepth, state, alpha, beta):
        v = None
        a = None
        # if we've reached the max depth
        if searchDepth == self.depth * state.getNumAgents():
            return self.evaluationFunction(state), None

        for action in state.getLegalActions(self.index):
            child = state.generateSuccessor(self.index, action)
            childV = self.minValueAction(searchDepth + 1, child, alpha, beta)[0]
            # set v to the max of v and childV
            if not v or childV > v:
                v = childV
                a = action
            # prune of v>beta
            if not beta or v > beta:
                return v, action
            alpha = max(v, alpha) if alpha else v
        # if we're at a leaf
        if len(state.getLegalActions()) == 0:
            v = self.evaluationFunction(state)
        return v, a

    def minValueAction(self, searchDepth, state, alpha, beta):
        func = self.maxValueAction if (searchDepth - 1) % state.getNumAgents() == 0 else self.minValueAction
        v = None
        a = None
        # if we've reached the max depth
        if searchDepth == self.depth * state.getNumAgents():
            return self.evaluationFunction(state), None

        ghostIndex = searchDepth % state.getNumAgents()

        for action in state.getLegalActions(ghostIndex):
            child = state.generateSuccessor(ghostIndex, action)

            if searchDepth == self.depth * state.getNumAgents() - 1:
                childV = self.evaluationFunction(child)
            else:
                childV = func(searchDepth + 1, child, alpha, beta)[0]
            # set v to the min of v and childV
            if not v or childV < v:
                v = childV
                a = action
            # prune if v<alpha
            # TODO: check if the parent is a min or max node...
            if not alpha or v < alpha:
                return v, action
            beta = min(v, beta) if beta else v
        # if we're at a leaf
        if len(state.getLegalActions()) == 0:
            v = self.evaluationFunction(state)
        return v, a


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
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

