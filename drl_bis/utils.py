from itertools import count
import os

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from typing import Dict


def play_one_episode(agent, env, is_train: str, batch_size: int, scaler: StandardScaler) -> float:
    # note: after transforming states are already 1xD
    state = env.reset()
    state = scaler.transform([state])
    done: bool = False
    info: Dict[str, float] = dict()

    while not done:
        # print(f"State: {state}")
        action = agent.act(state)
        # print(f"Agent took action: {action}")
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])
        if is_train == 'train':
            agent.update_replay_memory(state, action, reward, next_state, done)
            agent.replay(batch_size)
        state = next_state
        env.stock_owned_history.append(env.stock_owned)
        positions = np.append(env.stock_price * env.stock_owned, env.cash_in_hand)
        positions = positions.astype(np.int32).reshape(1, -1)
        env.cash_positions_history = np.vstack((env.cash_positions_history, positions))
        # print(f"Stocks owned: {env.stock_owned} + ${env.cash_in_hand:.0f} (bank account)")
        # print(f"Cash positions: {positions}")

    return float(info['cur_val'])


class ReplayBuffer:
    """
    The experience replay memory.

    Attributes
    ----------
    obs1_buf : numpy.ndarray
        Stores the states.

    obs2_buf : numpy.ndarray
        Stores the next states.

    acts_buf : numpy.ndarray
        Stores the actions. These are represented by integers going from zero up to N * 2 + 1,
        where N is the number of stocks.

    rews_buf : numpy.ndarray
        Stores the rewards.

    done_buf : numpy.ndarray
        Stores the done flag. The values can only be 0 or 1.

    pointer : int
        Pointer that starts at 0.

    size : int
        The current size of the buffer.

    max_size : int
        Maximum size of the buffer.
    """

    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.uint8)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.uint8)
        self.pointer: int = 0
        self.size: int = 0
        self.max_size: int = size

    def store(self, obs, action, reward: float, next_obs, done) -> None:
        p: int = self.pointer
        self.obs1_buf[p] = obs
        self.obs2_buf[p] = next_obs
        self.acts_buf[p] = action
        self.rews_buf[p] = reward
        self.done_buf[p] = done
        self.pointer = (p + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int = 32) -> Dict[str, np.ndarray]:
        """Chooses random indices from 0 up to the size of the buffer."""
        idxs: np.ndarray = np.random.randint(0, self.size, size=batch_size)
        batch: Dict[str, np.ndarray] = dict()
        batch['s'] = self.obs1_buf[idxs]
        batch['s2'] = self.obs2_buf[idxs]
        batch['a'] = self.acts_buf[idxs]
        batch['r'] = self.rews_buf[idxs]
        batch['d'] = self.done_buf[idxs]
        return batch


def get_scaler(env) -> StandardScaler:
    """

    :param env:
    :return:
    """
    # return scikit-learn scaler object to scale the states
    # Note: you could also populate the replay buffer here

    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


def check_dir(directory) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
        p = os.path.abspath(os.path.normpath(directory))
        print(f"Created directory: {p}")


def mlp(input_dim: int, n_action: int, n_hidden_layers: int = 1, hidden_dim: int = 32, verbose: int = 1) -> Model:
    """Create a neural network model (multi-layer perceptron). This is
    the model that will be used and trained by the Agent."""
    i = Input(shape=(input_dim, ))
    x = i

    for _ in range(n_hidden_layers):
        x = Dense(hidden_dim, activation='relu')(x)

    # Output layer
    x = Dense(n_action)(x)

    model = Model(i, x)

    model.compile(loss='mse', optimizer='adam')
    if verbose == 1:
        print(model.summary())

    return model


def elapsed_time(start: float, end: float) -> str:
    elapsed_secs: int = round(end - start)

    if elapsed_secs > 59:
        minutes: int = int(elapsed_secs / 60)
        seconds_msg: str = ' and ' + str(elapsed_secs % 60) + ' seconds' if elapsed_secs % 60 != 0 else ''
        return f"Elapsed time: {minutes} {'minutes' if minutes > 1 else 'minute'}{seconds_msg}"
    else:
        return f"Elapsed time: {elapsed_secs} seconds"
