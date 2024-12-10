from util import manhattanDistance
from game import Directions
import random, util

from game import Agent



class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
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
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
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

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def minimax(self, agentIndex, depth, gameState):
      # check terminal state or depth limit
      if gameState.isWin() or gameState.isLose() or depth <= 0:
          return self.evaluationFunction(gameState)
      
      nextAgent = (agentIndex + 1) % gameState.getNumAgents()
      nextDepth = depth - 1 if nextAgent == 0 else depth
      
      if agentIndex == 0: # pacman (maximizer)
          return max(self.minimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in gameState.getLegalActions(agentIndex))
      else: # ghost (minimizer)
          return min(self.minimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in gameState.getLegalActions(agentIndex))

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    # BEGIN_YOUR_ANSWER
    bestScore = float('-inf')
    bestAction = None
    for action in gameState.getLegalActions(0):
        score = self.minimax(1, 1, gameState.generateSuccessor(0, action))
        if score > bestScore:
            bestScore = score
            bestAction = action
    return bestAction
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER
    successorGameState = gameState.generateSuccessor(0, action)
    return self.minimax(1, 1, successorGameState)
    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
  """

  def expectimax(self, agentIndex, depth, gameState):
      # check terminal state or depth limit
      if gameState.isWin() or gameState.isLose() or depth <= 0:
          return self.evaluationFunction(gameState)
      
      legalActions = gameState.getLegalActions(agentIndex)
      if not legalActions:
          return self.evaluationFunction(gameState)
      
      nextAgent = (agentIndex + 1) % gameState.getNumAgents()
      nextDepth = depth - 1 if nextAgent == 0 else depth
      
      if agentIndex == 0: # pacman (maximizer)
          return max(self.expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in legalActions)
      else: # ghost (expected utility)
          return sum(self.expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in legalActions) / len(legalActions)

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    # BEGIN_YOUR_ANSWER
    # calc best action
    bestScore = float('-inf')
    bestAction = None
    for action in gameState.getLegalActions(0):
        score = self.expectimax(1, 1, gameState.generateSuccessor(0, action))
        if score > bestScore:
            bestScore = score
            bestAction = action
    return bestAction
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    successorGameState = gameState.generateSuccessor(0, action)
    return self.expectimax(1, 1, successorGameState)
    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing biased-expectimax

class BiasedExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your biased-expectimax agent (problem 3)
  """

  def biased_expectimax(self, agentIndex, depth, gameState):
      # check terminal state or depth limit
      if gameState.isWin() or gameState.isLose() or depth <= 0:
          return self.evaluationFunction(gameState)
      
      legalActions = gameState.getLegalActions(agentIndex)
      if not legalActions:
          return self.evaluationFunction(gameState)
      
      nextAgent = (agentIndex + 1) % gameState.getNumAgents()
      nextDepth = depth - 1 if nextAgent == 0 else depth
      
      if agentIndex == 0: # pacman (maximizer)
          return max(self.biased_expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in legalActions)
      else: # ghost (expected utility)
          probs = self.getProbs(legalActions)
          return sum(probs[action] * self.biased_expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in legalActions)

  def getProbs(self, actions): # from doc
        probs = {action: 0.5 / len(actions) for action in actions} # 0.5/len(actions) for every action
        if Directions.STOP in actions: # aaand add 0.5 for every STOP action
            probs[Directions.STOP] += 0.5
        return probs
  
  def getAction(self, gameState):
    """
      Returns the biased-expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing stop-biasedly from their
      legal moves.
    """
    # BEGIN_YOUR_ANSWER
    # calc best action
    bestScore = float('-inf')
    bestAction = None
    for action in gameState.getLegalActions(0):
        score = self.biased_expectimax(1, 1, gameState.generateSuccessor(0, action))
        if score > bestScore:
            bestScore = score
            bestAction = action
    return bestAction
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    successorGameState = gameState.generateSuccessor(0, action)
    return self.biased_expectimax(1, 1, successorGameState)
    # END_YOUR_ANSWER

######################################################################################
# Problem 4a: implementing expectiminimax

class ExpectiminimaxAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent (problem 4)
  """

  def expectiminimax(self, agentIndex, depth, gameState):
      # check terminal state or depth limit
      if gameState.isWin() or gameState.isLose() or depth <= 0:
          return self.evaluationFunction(gameState)
      
      legalActions = gameState.getLegalActions(agentIndex)
      if not legalActions:
          return self.evaluationFunction(gameState)
      
      nextAgent = (agentIndex + 1) % gameState.getNumAgents()
      nextDepth = depth - 1 if nextAgent == 0 else depth
      
      expectiminimaxRes = [self.expectiminimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in legalActions]
      if agentIndex == 0: # pacman (maximizer)
          return max(expectiminimaxRes)
      elif agentIndex % 2 == 1: # odd-numbered ghost (minimizer)
          return min(expectiminimaxRes)
      else: # even-numbered ghost (expected utility)
          return sum(expectiminimaxRes) / len(legalActions)
      
  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction

      The even-numbered ghost should be modeled as choosing uniformly at random from their
      legal moves.
    """
    # BEGIN_YOUR_ANSWER
    # calc best action
    bestScore = float('-inf')
    bestAction = None
    for action in gameState.getLegalActions(0):
        score = self.expectiminimax(1, 1, gameState.generateSuccessor(0, action))
        if score > bestScore:
            bestScore = score
            bestAction = action
    return bestAction
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    successorGameState = gameState.generateSuccessor(0, action)
    return self.expectiminimax(1, 1, successorGameState)
    # END_YOUR_ANSWER

######################################################################################
# Problem 5a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent with alpha-beta pruning (problem 5)
  """

  def alphaBeta(self, agentIndex, depth, gameState, alpha, beta):
    if gameState.isWin() or gameState.isLose() or depth <= 0:
        return self.evaluationFunction(gameState)

    legalActions = gameState.getLegalActions(agentIndex)
    if not legalActions:
        return self.evaluationFunction(gameState)

    nextAgent = (agentIndex + 1) % gameState.getNumAgents()
    nextDepth = depth - 1 if nextAgent == 0 else depth

    if agentIndex == 0: # pacman (maximizer)
        value = float('-inf')
        for action in legalActions:
            value = max(value, self.alphaBeta(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    elif agentIndex % 2 == 1: # odd-numbered ghost (minimizer)
        value = float('inf')
        for action in legalActions:
            value = min(value, self.alphaBeta(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value
    else: # even-numbered ghost (expected utility)
        total = 0
        for action in legalActions:
            total += self.alphaBeta(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action), alpha, beta)
        return total / len(legalActions)

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """
    # BEGIN_YOUR_ANSWER
    # calc best action
    bestScore = float('-inf')
    bestAction = None
    alpha = float('-inf')
    beta = float('inf')
    for action in gameState.getLegalActions(0):
        score = self.alphaBeta(1, 1, gameState.generateSuccessor(0, action), alpha, beta)
        if score > bestScore:
            bestScore = score
            bestAction = action
    return bestAction
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    successorGameState = gameState.generateSuccessor(0, action)
    alpha = float('-inf')
    beta = float('inf')
    return self.alphaBeta(1, 1, successorGameState, alpha, beta)
    # END_YOUR_ANSWER

######################################################################################
# Problem 6a: creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 6).
  """
  # BEGIN_YOUR_ANSWER
  def dist(a, b):
      _DIMENSION = 1
      if _DIMENSION == 1:
          return abs(a[0] - b[0]) + abs(a[1] - b[1])
      elif _DIMENSION == 2:
          return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
  
  posPacman = currentGameState.getPacmanPosition()
  foods = currentGameState.getFood().asList()
  capsules = currentGameState.getCapsules()
  scaredGhosts = [ghost for ghost in currentGameState.getGhostStates() if ghost.scaredTimer]
  normalGhosts = [ghost for ghost in currentGameState.getGhostStates() if not ghost.scaredTimer]

  # 1. get closer to food
  distFood = [dist(posPacman, food) for food in foods]
  distFoodMin = min(distFood) if distFood else 0

  # 2. get closer to capsule
  distCapsule = [dist(posPacman, capsule) for capsule in capsules]
  distCapsuleMin = min(distCapsule) if distCapsule else 0

  # 3. get closer to scared ghosts
  distScaredGhost = [dist(posPacman, scaredGhost.getPosition()) for scaredGhost in scaredGhosts]
  distScaredGhostMin = min(distScaredGhost) if distScaredGhost else 0

  # 4. eat food
  numRemainFood = len(foods) + 1

  # 5. eat capsule
  numRemainCapsule = len(capsules) + 1

  # 6. eat scared ghosts
  numRemainScaredGhost = len(scaredGhosts) + 1

  # 7. get far from normal ghosts
  distGhost = [dist(posPacman, ghost.getPosition()) for ghost in normalGhosts]
  distGhostMin = (min(distGhost) if distGhost else 0) + 1
  
  # smaller w @ phi -> better
  # w = [1.5, 0, 2, 4, 20, 4, 2]
  w = [1, 0, 1.5, 4, 20, 4, 0]
  bw = [1, 0, 1.5, 4, 20, 4, 0] # 1339
  w = bw
  phi = [
    distFoodMin, distCapsuleMin, distScaredGhostMin, 
    numRemainFood, numRemainCapsule, 1.0 / numRemainScaredGhost,
    1.0 / distGhostMin
  ]
  
  return currentGameState.getScore() - sum(w * phi for w, phi in zip(w, phi))
  # END_YOUR_ANSWER

def choiceAgent():
  """
    Choose the pacman agent model you want for problem 6.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
  """
  # BEGIN_YOUR_ANSWER
  return 'ExpectiminimaxAgent' # 'ExpectimaxAgent'
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction
