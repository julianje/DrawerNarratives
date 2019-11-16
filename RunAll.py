import numpy as np
import Observer
import sys
import csv

CulturalModel = True

# Part 1: LOAD WORLD AND AGENT MODELS
#####################################
ProgressBars = True
DrawerDimensions = [4, 4] # Rows followed by colum ns
Rationality = 0.01
# Build filenames
if CulturalModel:
	Basefile = 'POMDPs/Drawer'+str(DrawerDimensions[0])+'x'+str(DrawerDimensions[1])+'space_cultural'
else:
	Basefile = 'POMDPs/Drawer'+str(DrawerDimensions[0])+'x'+str(DrawerDimensions[1])+'space_rational'
WorldModel = Basefile+'.POMDP'
AgentModel = Basefile+'.policy'
OpenDrawers = []

sys.stdout.write("Loading model and policy...\n")
Observer = Observer.Observer(WorldModel, AgentModel, OpenDrawers, DrawerDimensions)
Observer.load()

Outputfile = 'DrawerPredictions.csv'

# Part 2: LOAD EACH TRIAL AND RUN
#################################
WriteHeader = True
with open('DrawerInputData_Exp.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=",")
	next(csv_reader) # skip header
	for trial in csv_reader:
		TrialName = trial[0]
		sys.stdout.write("Running trial: "+TrialName+"...\n")
		Observer.OpenDrawers = trial[1].split(' ')
		DrawerColors = trial[2]
		if DrawerColors == "":
			Observer.DrawerColors = np.zeros((DrawerDimensions[0], DrawerDimensions[1]))
		else:
			DrawerColors = [int(x) for x in DrawerColors.split(' ')]
			Observer.DrawerColors = np.reshape(DrawerColors, (DrawerDimensions[0], DrawerDimensions[1]))
		dumboptimize = int(trial[3]) # Dumboptimize takes a percentage and it only considered that percentage of paths sorted by their efficiency
		# This helps in just not considering paths that are too inefficient and would therefore have probabiliy 0 otherwise
		Observer.InferKnowledge(Rationality, progressbar = ProgressBars, dumboptimize = dumboptimize, cultural = CulturalModel)
		Observer.NormalizePosterior(rounded=True)
		if WriteHeader:
			Observer.SaveResults(filename=Outputfile, tid=TrialName, header=True)
			WriteHeader = False
		else:
			Observer.SaveResults(filename=Outputfile, tid=TrialName, header=False)
