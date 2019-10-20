import numpy as np
import Observer
import sys
# THINGS MISSING:
# 1. REMEMBERING IN BETWEEN
# 2. COLOR INFORMATION
# 3. In memory, we can do noisy memory around a drawer
# so middle doesn't quite work.

# TO DO:
# Fix memory hypotheses. Add first a filter abut which ones are allowed to be "memory centers"
# And fix the memory decay


# CONVENTIONS:
# First entry is always row and second is column for drawers

# Part 1: Set model parameters
DrawerDimensions = [3, 4] # 3 rows
Rationality = 0.01

# Build filenames
Basefile = 'Drawer'+str(DrawerDimensions[0])+'x'+str(DrawerDimensions[1])+'space'
WorldModel = Basefile+'.POMDP'
AgentModel = Basefile+'.policy'

# open one drawer in the middle
#OpenDrawers = ['m1-2']

# open top left corner
#OpenDrawers = ['m0-0','m0-1', 'm1-0']

# open first row and then go down
#OpenDrawers = ['m0-0','m0-1','m0-2','m1-2']

# open top left, go right, and go down
#OpenDrawers = ['m0-0','m0-1','m1-1']

# top left, middle left
#OpenDrawers = ['m0-0','m1-0']

# open first row
OpenDrawers = ['m0-0','m0-1', 'm0-2']

# open two drawers in the middle
#OpenDrawers = ['m2-3','m2-2']

# Part 2: Load model and run inference
sys.stdout.write("Loading model and policy...\n")
Observer = Observer.Observer(WorldModel, AgentModel, OpenDrawers, DrawerDimensions)
Observer.load()
sys.stdout.write("Inferring knowledge...\n")
Observer.InferKnowledge(Rationality)

# Part 3: Process posterior distribution and visualize
Results = Observer.ProcessPosterior(round=True)
Observer.PrintPosterior()
#Observer.PlotPosterior(Results, Title='_'.join(OpenDrawers))
