from JuniaDotsnBoxes import Env
import numpy as np
import time

env = Env()

#qtable = np.random.rand(env.state_count, env.actions_count).tolist()
qtable = np.load("qtable_size_4.npy").tolist()
print(qtable[0:10])

# 0 probability is line already drawn
# for i in range(env.state_count):
#     for j in range(env.number_of_lines):
#         if env.possible_states[i][j] == 1:
#             qtable[i][j] = 0

epochs = 1000
gamma = 0.1
epsilon = 0.08
decay = 0.1
lr = 0.07

player1_turn = True

victory = [0,0,0]

for i in range(epochs):
    print("epoch #", i + 1, "/", epochs)
    state, reward, done = env.reset()

    while not done:
        if player1_turn:
            action = env.randomAction()
            next_state, reward, done, player1_turn = env.step(action, player1_turn)
        else :
            if np.random.uniform() < epsilon:
                action = env.randomAction()
            else:
                action = qtable[state].index(max(qtable[state]))
                if (env.init_moves_remaining[action] in env.moves_remaining) == False:
                    qtable[state][action] = 0 # 0 probability is line already drawn
                    action = env.randomAction()
                   
            next_state, reward, done, player1_turn = env.step(action, player1_turn)

            qtable[state][action] = qtable[state][action] + lr * (reward + gamma * max(qtable[next_state]))

        state = next_state
    
    epsilon -= decay * epsilon

    if env.score[0] > env.score[1]:
        victory[0] += 1
        print(f"Random won! Score: {env.score[0]} to {env.score[1]}")
    elif env.score[1] > env.score[0]:
        victory[1] += 1
        print(f"RL won! Score: {env.score[0]} to {env.score[1]}")
    else:
        victory[2] += 1
        print(f"It's a draw! Score: {env.score[0]} to {env.score[1]}")
        

print(f"\nNombre de victoire du random : {victory[0]}")
print(f"Nombre de victoire de rl : {victory[1]}")
print(f"Nombre de draw : {victory[2]}")

np.save('qtable_size_4.npy', qtable)