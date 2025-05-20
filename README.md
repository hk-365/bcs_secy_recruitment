# Triwizard Maze Challenge - DQN Implementation

## Environment assumptions:
- have used first maze txt out of the given three.
- Harry and death eater both move with same speed.
- Harry has ability to move 2 spots at a time with 10% probability.
- Death eater strictly follows bfs algorithm.
- State is defined by 10 scalars which include harry's absolute position, harry's relative position wrt the cup, harry's relative position wrt the death eater, wall proximity indicators.
- Reward structure:
    - +100 if harry reaches the cup
    - -50 if harry is caught by the death eater.
    - rewarded if harry get's closer to the wall in the way +3*(old dist with cup-new dist with cup)
    - penalised if harry get's closer to the death eater in th way -1*(old dist with death eater - new dist with death eater)
    - -0.5 penalty for every step
  
## Approach
- Implemented Deep Q-Network (DQN) with replay buffer, policy network and target network
- State representation includes positions and relative distances
- Reward shaping to encourage reaching the cup and avoiding the Death Eater

## Evaluation metrics
- During training: no. of generations needed for harry to escape 10 times in a row: 284
- The graph of Moving avg (over 100 episodes) is saved as training_metrics_moving_avg.png in the repo.
- The graph of Success rate (over 50 episodes) is saved as training_metrics_success_rate.png in the repo.
- Final success rate during training: 69.36%
- Final trained weights are saved as dqn_weights.pth in "models" directory.
  
## How to run using trained weights
- clone the repo
- type in cmd: python run_trained.py
- It will run for a total of 10 episodes
- In the end you will see the final success rate
- The pygame window will open
      - Blue square: Represents Harry
      - Red square: Represents Death eater
      - Golden square: Represents the Cup
      - Grey boxes: Represent the walls
