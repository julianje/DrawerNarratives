import numpy as np
import Observer

# CONVENTIONS:
# First entry is always row and second is column for drawers

# Part 1: Set model parameters
WorldModel = 'Drawer3x4space.POMDP'
AgentModel = 'Drawer3x4space.policy'
DrawerDimensions = [3, 4] # 3 rows
OpenDrawers = ['m0-0','m0-1','m0-2']

# Part 2: Load model and run inference
Observer = Observer.Observer(WorldModel,AgentModel,OpenDrawers, DrawerDimensions)
Observer.load()
Observer.InferKnowledge()

# Part 3: Process posterior distribution and visualize

