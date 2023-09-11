import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


def initialize():
    # Initialize environment
    env = gym.make("CartPole-v1")

    # Wrap environment
    env = DummyVecEnv([lambda: env])

    # Create Logs directory
    log_path = os.path.join("Training", "Logs")

    # Create model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

    # Create directory to save model
    ppo_path = os.path.join("Training", "Saved Models", "PPO_Model_Cartpole")

    # Save model
    model.save(ppo_path)

    return env


def train(ppo_path, env, timesteps):
    # Load model
    model = PPO.load(ppo_path, env=env)

    # Train model
    model.learn(total_timesteps=timesteps, progress_bar=True)

    # Save model
    model.save(ppo_path)


def evaluate(ppo_path, env):
    # Load model
    model = PPO.load(ppo_path, env=env)
    mean_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    print(mean_reward)


def test(ppo_path):
    # Load model
    model = PPO.load(ppo_path)
    new_env = gym.make("CartPole-v1", render_mode="human")
    episodes = 5
    for episode in range(1, episodes + 1):
        obs, _ = new_env.reset()
        done = False
        score = 0

        steps = 0
        while not done:
            if steps > 1000:
                break
            new_env.render()
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = new_env.step(action)
            score += reward
            steps += 1
        print(f"Episode {episode}: Score: {score}")
    new_env.close()
