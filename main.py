import os

import torch.utils.tensorboard

import ppo_model

if __name__ == '__main__':
    path = os.path.join("Training", "Saved Models", "PPO_Model_Cartpole")
    env = ppo_model.initialize()
    ppo_model.train(path, env, timesteps=1)
    ppo_model.evaluate(path, env)
    ppo_model.test(path)

    ppo_model.train(path, env, timesteps=10)
    ppo_model.evaluate(path, env)
    ppo_model.test(path)

    ppo_model.train(path, env, timesteps=50)
    ppo_model.evaluate(path, env)
    ppo_model.test(path)

    ppo_model.train(path, env, timesteps=1000)
    ppo_model.evaluate(path, env)
    ppo_model.test(path)

    ppo_model.train(path, env, timesteps=100000)
    ppo_model.evaluate(path, env)
    ppo_model.test(path)

    env.close()
