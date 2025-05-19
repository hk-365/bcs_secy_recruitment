import pygame
import time
from utils.maze_env import MazeEnv
from utils.dqn_agent import DQNAgent

pygame.init()
pygame.display.set_caption("Triwizard Tournament - Maze Challenge")

def run_trained_model(episodes=10, render=True, delay=200):
    # Initialize environment and agent
    env = MazeEnv()
    state_size = len(env._get_state())
    action_size = len(ACTIONS)
    agent = DQNAgent(state_size, action_size)
    
    # Load trained weights
    agent.load_model("models/dqn_weights.pth")
    agent.epsilon = 0.01  # Minimal exploration
    
    success_count = 0
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            if render:
                env.render()
                pygame.time.delay(delay)
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
            
            # Get action from trained model
            action_idx = agent.act(state)
            action = ACTIONS[action_idx]
            
            # Take action
            next_state, reward, done = env.step(action)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if render:
                pygame.display.set_caption(
                    f"Episode: {e+1} | Steps: {steps} | "
                    f"Reward: {total_reward:.1f} | Action: {action}"
                )
        
        if reward > 90:
            success_count += 1
            result = "SUCCESS!"
        else:
            result = "FAILED!"
        
        print(f"Episode {e+1}: {result} Steps: {steps}, Total Reward: {total_reward:.1f}")
    
    print(f"\nSuccess rate: {success_count/episodes*100:.2f}%")
    pygame.quit()

if __name__ == "__main__":
    run_trained_model(
        episodes=10,    # Number of episodes to run
        render=True,    # Set to False for faster evaluation
        delay=200       # Visualization speed (ms)
    )