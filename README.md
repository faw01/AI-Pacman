![Pacman Game](pacman_game.gif)

# AI-Pacman

This repository contains implementations for Monash University's FIT3080 Artificial Intelligence unit. These assignments centered around developing AI agents for the classic Pacman game. The project uses the UC Berkeley Pacman framework and includes various AI techniques from search algorithms to reinforcement learning and machine learning.

## Overview

The repository includes two main assignments:

- **Assignment 1**: Search Algorithms
  - Single-agent search using A* algorithm
  - Adversarial search using Alpha-Beta pruning

- **Assignment 3**: Reinforcement Learning & Machine Learning
  - Value Iteration for Markov Decision Processes
  - Q-Learning with epsilon-greedy exploration
  - Perceptron-based supervised learning

## Assignment 1: Search

### Components

#### 1. Single-Agent Search (Q1)

##### Q1a: A* with Manhattan Distance

Implements A* search with Manhattan distance heuristic to find optimal paths to a single food dot.

**Implementation Highlights:**
```python
def q1a_solver(problem):
    # Initialize A* data structures
    frontier = util.PriorityQueue()
    explored = set()
    start_state = problem.getStartState()
    goal = problem.getGoalState()
    
    # Add start state to frontier with priority = heuristic(start, goal)
    frontier.push((start_state, []), manhattanDistance(start_state, goal))
    
    while not frontier.isEmpty():
        current, path = frontier.pop()
        
        if problem.isGoalState(current):
            return path
            
        if current not in explored:
            explored.add(current)
            
            for successor, action, _ in problem.getSuccessors(current):
                if successor not in explored:
                    new_path = path + [action]
                    priority = len(new_path) + manhattanDistance(successor, goal)
                    frontier.push((successor, new_path), priority)
    
    return []
```

**Key Findings:**
- The A* algorithm consistently found optimal paths across all maze sizes
- Main limitation observed was in mazes with tight corridors and dead ends
- The Manhattan distance heuristic sometimes led A* to explore directions that were ultimately blocked

##### Q1b: Finding the Closest Dot

Extends the A* algorithm to navigate to the closest dot in the maze.

**Implementation Highlights:**
```python
# Heuristic function to find closest food
def heuristic(state, foodGrid):
    x, y, _ = state
    food_locations = foodGrid
    
    if not food_locations:
        return 0
        
    distances = [manhattanDistance((x, y), food) for food in food_locations]
    return min(distances)

# Tie-breaking mechanism in A*
def astar_with_tiebreaking(problem):
    frontier = util.PriorityQueue()
    explored = set()
    
    # Use tuple (f, h) as priority for tie-breaking
    for successor, action, _ in problem.getSuccessors(current):
        if successor not in explored:
            g = len(new_path)
            h = heuristic(successor, goal)
            f = g + h
            priority = (f, h)  # Tie-breaking tuple
            frontier.push((successor, new_path), priority)
```

**Key Findings:**
- The tie-breaking mechanism significantly improved performance by preferring states closer to food when f-values were equal
- Surprisingly, simple heuristics often outperformed complex ones due to computational overhead
- More sophisticated heuristics including BFS-based, MST, and combinations of distances did not significantly improve performance

##### Q1c: Collecting Multiple Dots

Uses a greedy best-first search approach to efficiently collect multiple dots while maximizing score.

**Implementation Highlights:**
```python
def q1c_solver(problem):
    # Using greedy best-first search instead of A*
    frontier = util.PriorityQueue()
    explored = set()
    start_state = problem.getStartState()
    
    frontier.push((start_state, []), len(start_state[1]))  # Priority = number of remaining food
    
    while not frontier.isEmpty():
        current, path = frontier.pop()
        
        if current in explored:
            continue
            
        if problem.isGoalState(current):
            return path
            
        explored.add(current)
        
        for successor, action, _ in problem.getSuccessors(current):
            if successor not in explored:
                new_path = path + [action]
                # Simple heuristic: number of remaining food dots
                priority = len(successor[1])
                frontier.push((successor, new_path), priority)
    
    return []
```

**Key Findings:**
- Greedy best-first search significantly outperformed A* for this problem
- The A* algorithm struggled with the larger state space, resulting in negative scores
- The concept of "reachable food" effectively reduced the state space and improved performance
- Simple remaining food count heuristic provided excellent results

#### 2. Adversarial Search (Q2)

Implements Alpha-Beta pruning for playing complete Pacman games against ghosts.

**Implementation Highlights:**
```python
def alpha_beta_search(self, gameState):
    def max_value(state, alpha, beta, depth):
        if self.isTerminal(state, depth):
            return self.evaluateState(state), None
            
        v, move = float("-inf"), None
        
        for action in state.getLegalActions(0):
            successor = state.generateSuccessor(0, action)
            v2, _ = min_value(successor, alpha, beta, depth, 1)
            
            if v2 > v:
                v, move = v2, action
                
            if v >= beta:
                return v, move
                
            alpha = max(alpha, v)
            
        return v, move
        
    def min_value(state, alpha, beta, depth, agent_index):
        if self.isTerminal(state, depth):
            return self.evaluateState(state), None
            
        v, move = float("inf"), None
        next_agent = (agent_index + 1) % state.getNumAgents()
        
        if next_agent == 0:
            depth += 1
            
        for action in state.getLegalActions(agent_index):
            successor = state.generateSuccessor(agent_index, action)
            
            if next_agent == 0:
                v2, _ = max_value(successor, alpha, beta, depth)
            else:
                v2, _ = min_value(successor, alpha, beta, depth, next_agent)
                
            if v2 < v:
                v, move = v2, action
                
            if v <= alpha:
                return v, move
                
            beta = min(beta, v)
            
        return v, move
    
    _, action = max_value(gameState, float("-inf"), float("inf"), 0)
    return action
```

**Evaluation Function:**
```python
def evaluateState(self, state):
    if state.isLose():
        return -500 + state.getScore()
    if state.isWin():
        return 500 + state.getScore()
        
    position = state.getPacmanPosition()
    food = state.getFood().asList()
    
    # Find closest food using BFS
    foodDist = self.bfsDistance(position, food, state)
    if foodDist == float("inf"):
        foodDist = min(manhattanDistance(position, f) for f in food)
    foodScore = 1.0 / (foodDist + 1)
    
    # Ghost avoidance
    ghostDist = self.findMinimumGhostDistance(position, state.getGhostStates())
    ghostPenalty = -200 if ghostDist < 2 else 0
    
    return (ghostPenalty) + (foodScore) + state.getScore()
```

**Key Findings:**
- A search depth of 2 provided the best balance between performance and computational efficiency
- Ghost avoidance was the most challenging aspect to optimize
- Simple evaluation functions often outperformed more complex ones
- Binary ghost avoidance (applying penalty within critical distance) was more effective than continuous distance-based penalties

### Running Assignment 1

**Single Agent Search (Q1):**

```
# Q1a: A* to a single dot
python pacman.py -l layouts/q1a_tinyMaze.lay -p SearchAgent -a fn=q1a_solver,prob=q1a_problem --timeout=1

# Q1b: A* to the closest dot
python pacman.py -l layouts/q1b_tinyCorners.lay -p SearchAgent -a fn=q1b_solver,prob=q1b_problem --timeout=5

# Q1c: Collect multiple dots
python pacman.py -l layouts/q1c_tinySearch.lay -p SearchAgent -a fn=q1c_solver,prob=q1c_problem --timeout=10
```

**Adversarial Search (Q2):**

```
# Alpha-Beta pruning for complete Pacman games
python pacman.py -l layouts/q2_testClassic.lay -p Q2_Agent --timeout=30
```

## Assignment 3: Reinforcement Learning & Machine Learning

### Components

#### 1. Value Iteration (Q1)

Implementation of value iteration for MDPs in the Pacman environment with stochastic movements (80% success, 10% slip left, 10% slip right).

**Implementation Highlights:**
```python
def registerInitialState(self, gameState):
    self.MDP = self.mdp_func(gameState)
    self.values = np.zeros((self.MDP.grid_width, self.MDP.grid_height))
    
    for _ in range(self.iterations):
        updated_values = np.zeros_like(self.values)
        for x in range(self.MDP.grid_width):
            for y in range(self.MDP.grid_height):
                current_state = (x, y)
                if self.MDP.isTerminal(current_state):
                    updated_values[x, y] = self.MDP.getReward(current_state, None, None)
                else:
                    legal_actions = self.MDP.getPossibleActions(current_state)
                    if legal_actions:
                        updated_values[x, y] = max(
                            self.computeQValueFromValues(current_state, action)
                            for action in legal_actions
                        )
        self.values = updated_values

def computeQValueFromValues(self, state, action):
    q_value = 0.0
    for next_state, prob in self.MDP.getTransitionStatesAndProbs(state, action):
        immediate_reward = self.MDP.getReward(state, action, next_state)
        future_value = self.discount * self.values[next_state[0]][next_state[1]]
        q_value += prob * (immediate_reward + future_value)
    return q_value
```

**Optimal Parameters:**
- Small mazes: γ = 0.97
- Medium mazes: γ = 0.99
- Large mazes: γ = 0.99

**Key Findings:**
- Small maze performance exhibited a clear optimal point at γ = 0.97, with performance plateauing beyond that
- Medium maze performance demonstrated a more linear improvement trajectory with increasing gamma values
- Large maze performance was most sensitive to gamma values, with some layouts showing dramatic improvements

#### 2. Q-Learning (Q2)

Q-learning implementation with epsilon-greedy action selection for non-deterministic environments.

**Implementation Highlights:**
```python
def getQValue(self, state, action):
    x, y = state.getPacmanPosition()
    action_index = self.getActionIndex(action)
    return self.Q_values[x, y, action_index]

def update(self, state, action, nextState, reward):
    x, y = state.getPacmanPosition()
    action_index = self.getActionIndex(action)
    
    current_q = self.getQValue(state, action)
    next_max_q = self.computeValueFromQValues(nextState)
    
    updated_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.discount * next_max_q)
    self.Q_values[x, y, action_index] = updated_q

def epsilonGreedyActionSelection(self, state):
    legal_actions = self.getLegalActions(state)
    if not legal_actions:
        return None
        
    if random.random() < self.epsilon:
        return random.choice(legal_actions)
    else:
        return self.computeActionFromQValues(state)
```

**Optimal Parameters:**
- Tiny mazes: ε = 0.1, α = 0.5, γ = 0.7
- Small mazes: ε = 0.2, α = 0.3, γ = 0.8
- Medium mazes: ε = 0.3, α = 0.2, γ = 0.9

**Key Findings:**
- Q-learning achieved perfect win rates across all maze sizes with tuned parameters
- Performance improved when exploration rate and learning rate decreased with increasing maze size
- Discount factor needed to increase with maze size to account for longer planning horizons
- Q-learning showed more robustness than value iteration across maze layouts

#### 3. Perceptron for Classic Pacman (Q3)

Single-layer perceptron model trained on gameplay data to predict optimal actions.

**Implementation Highlights:**
```python
def predict(self, features):
    # Calculate dot product and apply activation function
    z = np.dot(features, self.weights)
    return self.activationOutput(z)

def activationOutput(self, x):
    # Sigmoid activation function with clipping for stability
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def train(self, trainingData, trainingLabels, validationData, validationLabels):
    # Initialize weights randomly
    num_features = trainingData.shape[1]
    self.weights = np.random.randn(num_features) * 0.01
    
    # Track metrics
    training_losses = []
    validation_losses = []
    
    for iteration in range(self.max_iterations):
        # Forward pass
        predictions = np.array([self.predict(x) for x in trainingData])
        
        # Calculate loss
        epsilon = 1e-15  # Prevent log(0)
        loss = -np.mean(trainingLabels * np.log(predictions + epsilon) + 
                         (1 - trainingLabels) * np.log(1 - predictions + epsilon))
        
        # Calculate gradients
        errors = predictions - trainingLabels
        gradients = np.zeros_like(self.weights)
        
        for i in range(len(trainingData)):
            gradients += errors[i] * trainingData[i]
        
        gradients /= len(trainingData)
        
        # Update weights
        self.weights -= self.learning_rate * gradients
        
        # Evaluate on validation set
        if iteration % 10 == 0:
            # Calculate validation metrics
            val_predictions = np.array([self.predict(x) for x in validationData])
            val_loss = -np.mean(validationLabels * np.log(val_predictions + epsilon) + 
                               (1 - validationLabels) * np.log(1 - val_predictions + epsilon))
            
            training_losses.append(loss)
            validation_losses.append(val_loss)
    
    return training_losses, validation_losses
```

**Features Used:**
```python
feature_names_to_use = [
    'closestFood', 'closestFoodNow', 'closestGhost', 'closestGhostNow',
    'closestScaredGhost', 'closestScaredGhostNow', 'eatenByGhost',
    'eatsCapsule', 'eatsFood', "foodCount", 'foodWithinFiveSpaces',
    'foodWithinNineSpaces', 'foodWithinThreeSpaces', 'furthestFood',
    'numberAvailableActions', "ratioCapsuleDistance", "ratioFoodDistance",
    "ratioGhostDistance", "ratioScaredGhostDistance"
]
```

**Key Findings:**
- Most influential features (by weight magnitude):
  - ratioFoodDistance (-14.26): Getting closer to food is strongly preferred
  - eatsCapsule (7.95): Eating a capsule is highly rewarded
  - ratioScaredGhostDistance (-6.56): Getting closer to scared ghosts is preferred
  - closestGhost (-5.08): Staying away from normal ghosts is important
- A learning rate of 0.01 with 100 training epochs yielded the best performance (92.7% test accuracy)
- Removing even a single feature significantly decreased performance
- Feature importance aligned with intuitive game strategies (approach food/capsules, avoid ghosts, chase scared ghosts)

### Running Assignment 3

**Value Iteration (Q1):**

```
# Running Value Iteration
python pacman.py -l layouts/VI_smallMaze1_1.lay -p Q1Agent -a discount=gamma,iterations=K -g StationaryGhost -n 20
```

**Q-Learning (Q2):**

```
# Running Q-Learning
python pacman.py -l layouts/QL_tinyMaze1_1.lay -p Q2Agent -a epsilon=epsilon,alpha=alpha,gamma=gamma -x K -n N -g StationaryGhost
```

**Perceptron (Q3):**

```
# Training the perceptron
python trainPerceptron.py -i K -l alpha -w weight_save_path

# Running the trained perceptron
python pacman.py -l layouts/ML_mediumClassic.lay -p Q3Agent -a weights_path=weight_save_path
```

## Project Structure

- **a1/**: Assignment 1 files
  - **agents/**: Agent implementations
  - **layouts/**: Maze layouts for different problems
  - **problems/**: Problem definitions
  - **solvers/**: Search algorithm implementations

- **a3/**: Assignment 3 files
  - **agents/**: Agent implementations
  - **layouts/**: Maze layouts
  - **models/**: Trained models and parameters
  - **pacmandata/**: Training data for the perceptron

## Environment

The code runs in a Python environment with the following dependencies:
- Python 3.x
- NumPy
- SciPy (optional)
