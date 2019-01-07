import gym
import gym_gomoku

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Reshape, MaxPooling1D
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory


def create_model_cnn():
    model = Sequential()
    model.add(Reshape((env.board_size, env.board_size, 1), input_shape=(1, env.board_size, env.board_size)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))

    print(model.summary())
    return model


def train_model(env, model, learning_rate=1e-3, eps=.1, gamma=.99):
    memory = SequentialMemory(limit=100000, window_length=1)
    policy = EpsGreedyQPolicy(eps=eps)
    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=1,
                   target_model_update=1e-2, policy=policy, gamma=gamma)
    dqn.compile(Adam(lr=learning_rate), metrics=['mae'])
    dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2)


if __name__ == '__main__':
    env = gym.make('Gomoku9x9-v0')
    model = create_model_cnn()
    train_model(env, model)
