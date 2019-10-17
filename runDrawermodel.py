import numpy as np
import Observer
import sys
# THINGS MISSING:
# 1. REMEMBERING IN BETWEEN
# 2. COLOR INFORMATION


# CONVENTIONS:
# First entry is always row and second is column for drawers

# Part 1: Set model parameters
WorldModel = 'Drawer3x4space.POMDP'
AgentModel = 'Drawer3x4space.policy'
DrawerDimensions = [3, 4] # 3 rows

# open first row and then go down
#OpenDrawers = ['m0-0','m0-1','m0-2','m1-2']

# top left, middle left
#OpenDrawers = ['m0-0','m1-0']

# open first row
OpenDrawers = ['m0-0','m0-1', 'm0-2']

# open one drawer in the middle
#OpenDrawers = ['m2-3']

# open two drawers in the middle
#OpenDrawers = ['m2-3','m2-2']

# Part 2: Load model and run inference
sys.stdout.write("Loading model and policy...\n")
Observer = Observer.Observer(WorldModel, AgentModel, OpenDrawers, DrawerDimensions)
Observer.load()
Observer.InferKnowledge()

# Part 3: Process posterior distribution and visualize
Results = Observer.ProcessPosterior(round=True)
Observer.PlotPosterior(Results[1])
