import numpy as np
from .utils import ReplayBuffer, mlp


class DQNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(state_size, action_size, size=500)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = mlp(state_size, action_size)

    def update_replay_memory(self, state, action, reward, next_state, done) -> None:
        self.memory.store(state, action, reward, next_state, done)

    def act(self, state) -> int:
        """Get action the leads to the maximum Q-Value using epsilon-greedy.

        Returns
        -------
        int
            An action.
        """
        return self.__epsilon_greedy_policy(state)
        # if np.random.rand() <= self.epsilon:
        #     return np.random.choice(self.action_size)
        # act_values = self.model.predict(state)
        # return np.argmax(act_values[0])

    def __epsilon_greedy_policy(self, state) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            q_values = self.model.predict(state)
            return int(np.argmax(q_values[0]))

    def replay(self, batch_size=32) -> None:
        """Function that does the learning.

        Parameters
        ----------
        batch_size : int
            How many samples to grab from the replay memory.
        """
        if self.memory.size < batch_size:  # Check if enough data in memory
            return

        # sample a batch of data from the replay memory
        minibatch = self.memory.sample_batch(batch_size)
        states = minibatch['s']
        actions = minibatch['a']
        rewards = minibatch['r']
        next_states = minibatch['s2']
        done = minibatch['d']

        # Calculate the tentative target: Q(s',a)
        target = rewards + (1 - done) * self.gamma * np.amax(self.model.predict(next_states), axis=1)

        # With the Keras API, the target (usually) must have the same
        # shape as the predictions.
        # However, we only need to update the network for the actions
        # which were actually taken.
        # We can accomplish this by setting the target to be equal to
        # the prediction for all values.
        # Then, only change the targets for the actions taken.
        # Q(s,a)
        target_full = self.model.predict(states)
        target_full[np.arange(batch_size), actions] = target

        # Run one training step
        self.model.train_on_batch(states, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
