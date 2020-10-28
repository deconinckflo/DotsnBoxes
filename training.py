from JuniaDotsnBoxes import Env
from dqn import DQN
import numpy as np
from tqdm import tqdm
import time

# Environment settings
EPISODES = 1000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

SAVE_EVERY = 200
MODEL_NAME = "2x64"

env = Env()
agent = DQN()

player1_turn = True

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    step = 1

    # Reset environment and get initial state
    current_state, reward, done = env.reset()

    # Reset flag and start iterating until episode ends
    while not done:

        if player1_turn:
            action = env.randomAction()
        else :
            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
              # Get action from Q table
              action = np.argmax(agent.get_qs(current_state))
              if (env.init_moves_remaining[action] in env.moves_remaining) == False:
                    action = env.randomAction()
            else:
              # Get random action
              action = env.randomAction()

        new_state, reward, done, player1_turn = env.step(action, player1_turn)

        # Every step we update replay memory and train main network
        #print("memory")
        #print((current_state, action, reward, new_state, done))
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    if episode % SAVE_EVERY == 0:
      agent.model.save("models/" + MODEL_NAME + "_" + str(episode) + ".h5")
      
