from random import *
import time
from tensorflow.keras.models import load_model
import numpy as np
# Init your variables here 

# Put your name Here 
name = "Deconinck Florian"

# Coder son propre agent
print("-"*60)
model = load_model("models/2x64_1000.h5")


# C3PO strategy : Return the first element in the available cells.


def prepare_init_moves_remaining(size):
        moves_remaining = []
        for i in range(size**2):
            if (i+1) % size != 0:
                moves_remaining.append((i,i+1))
            if i+size < size**2:
                moves_remaining.append((i,i+size))
        return moves_remaining

init_moves_remaining = prepare_init_moves_remaining(4)

def play(board, available_cells, player):
    time.sleep(0.5)
    state = np.array([0 if x in available_cells else 1 for x in init_moves_remaining])
    action = np.argmax(model.predict(state.reshape(-1,*state.shape))[0])
    print(model.predict(state.reshape(-1,*state.shape))[0])
    if(init_moves_remaining[action] in available_cells):
        print("normal")
        return init_moves_remaining[action]
    else:
        return choice(available_cells)