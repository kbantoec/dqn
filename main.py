import pandas as pd
import numpy as np
import os
import argparse
import pickle
from time import time
from drl.utils import normabspath, make_dir_if_not, get_scaler, play_one_episode, show_elapsed_time
from drl.environment import MultiStockEnv
from drl.agent import DQNAgent

if __name__ == '__main__':
    basedir: str = os.path.dirname(__file__)
    models_folder: str = normabspath(basedir, 'models')
    rewards_folder: str = normabspath(basedir, 'rewards')
    num_episodes: int = 2_000
    batch_size: int = 32
    initial_investment: float = 20_000.0

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help="Modes: 'train' or 'test'.")
    parser.add_argument('-s', '--traintestsplit', type=float, required=False, help="Training set proportion.")
    args = parser.parse_args()

    make_dir_if_not(models_folder)
    make_dir_if_not(rewards_folder)

    data = np.loadtxt('data/aapl_msi_sbux.csv', delimiter=',', skiprows=1)

    n_timesteps, n_stocks = data.shape

    if args.traintestsplit:
        n_train = int(float(args.traintestsplit) * n_timesteps)
    else:
        n_train = int(0.85 * n_timesteps)

    train_data = data[:n_train]
    test_data = data[n_train:]

    env = MultiStockEnv(train_data, initial_investment)
    state_size: int = env.state_dim
    action_size: int = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)  # TODO: check

    portfolio_value: list = []

    if args.mode == 'test':
        # Load the scaler created upon training
        with open(f"{normabspath(models_folder, 'scaler.pkl')}", 'rb') as f:
            scaler = pickle.load(f)

        env = MultiStockEnv(test_data, initial_investment)

        agent.epsilon = 0.01

        # Load weights
        agent.load(f"{normabspath(models_folder, 'dqn.h5')}")

    for episode in range(num_episodes):
        start = time()
        val = play_one_episode(agent, env, args.mode, batch_size)
        print(f"Episode {episode + 1}/{num_episodes}, Portfolio end value: {val:.2f}")
        show_elapsed_time(start=start, end=time())

    if args.mode == 'train':
        agent.save(f"{normabspath(models_folder, 'dqn.h5')}")

        with open(f"{normabspath(models_folder, 'scaler.pkl')}", 'wb') as f:
            pickle.dump(scaler, f)

    np.save(f"{normabspath(rewards_folder, args.mode + '.npy')}", portfolio_value)



