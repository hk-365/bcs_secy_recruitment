import pygame
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.Maze_env import MazeEnv
from utils.dqn_agent import DQNAgent

pygame.init()
pygame.display.set_caption("Triwizard Tournament - Maze Challenge")

def train_agent():
    env = MazeEnv()
    state_size = len(env._get_state())
    action_size = len(ACTIONS)
    agent = DQNAgent(state_size, action_size)
    
    episodes = 10000
    batch_size = 32
    success_streak = 0
    success_threshold = 10
    rewards_history = []
    success_history = []
    x=episodes
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action_idx = agent.act(state)
            action = ACTIONS[action_idx]
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        #update target network periodically every 10 episodes
        if e%10== 0:
            agent.update_target_model()
        
        #track success
        if reward> 90:  # Harry reached the cup
            success_streak+= 1
        else:
            success_streak= 0
            
        rewards_history.append(total_reward)
        success_history.append(1 if success_streak> 0 else 0)
        
        if (e+1) % 500== 0:
            print(f"Episode: {e+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        if success_streak>= success_threshold:
            print(f"Harry escaped successfully {success_threshold} times in a row! in {e+1} generations")
            x=e+1
            break


    if x<=episodes:
        for e in range(episodes-x+1):
            state = env.reset()
            total_reward = 0
            done = False
        
            while not done:
              action_idx = agent.act(state)
              action = ACTIONS[action_idx]
              next_state, reward, done = env.step(action)
            
              agent.remember(state, action_idx, reward, next_state, done)
              state = next_state
              total_reward += reward
            
              if len(agent.memory) > batch_size:
                  agent.replay(batch_size)
        
            #update target network periodically after every 10 episodes
            if e % 10 == 0:
              agent.update_target_model()

            #track success
            if reward> 90:  # Harry reached the cup
              success_streak+= 1
            else:
              success_streak= 0
            
            rewards_history.append(total_reward)
            success_history.append(1 if success_streak > 0 else 0)
        
            if (e+1+x)% 500 == 0:
                print(f"Episode: {e+1+x}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    agent.save_model("models/dqn_weights.pth")

    #Moving average of rewards
    window_size= episodes// 100
    moving_avg= np.convolve(rewards_history, np.ones(window_size)/ window_size, mode='valid')

    #Plot
    plt.plot(moving_avg)
    plt.xlabel('Episode')
    plt.ylabel(f'Moving Average (window size {window_size})')
    plt.title('Moving Average Reward over Episodes')
    plt.grid(True)
    plt.show()

    #Succes rate
    window_size= 50
    success_rate = np.convolve(success_history, np.ones(window_size)/window_size, mode='valid')
    plt.plot(success_rate)
    plt.xlabel('Episode')
    plt.ylabel(f'Success Rate (moving avg over {window_size} episodes)')
    plt.title('Harry’s Escape Success Rate During Training')
    plt.grid(True)
    plt.savefig("training_metrics.png")
    plt.show()

    final_success_rate=(np.sum(success_history)/episodes)* 100
    print(f"Final Success Rate: {final_success_rate:.2f}%")
    
    #Save model
    torch.save(agent.model.state_dict(), "harry_potter_dqn.pth")

if __name__ == "__main__":
    train_agent()