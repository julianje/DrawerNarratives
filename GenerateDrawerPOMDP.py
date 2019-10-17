import sys
import csv
import numpy as np


def PrintList(listname, Inputlist, filewriter):
	"""
	Print a list with a name using filewriter.
	This is a supporting function that PrintPOMDP uses to create the .POMDP file
	"""
	filewriter.write(listname + ': ')
	Discard = [filewriter.write(str(Inputlist[x]) + ' ') for x in range(len(Inputlist)-1)]
	filewriter.write(str(Inputlist[len(Inputlist)-1]) + '\n')

def BuildPOMDP(TrialName, DrawerDimensions, PlanningReward = 50, FutureDiscount = 0.9, ActionCost = -1.0):
	# First mark all positions and flatten
	BaseStates = [[str(x)+"-"+str(y) for x in range(DrawerDimensions[0])] for y in range(DrawerDimensions[1])]
	BaseStates = [item for sublist in BaseStates for item in sublist]
	# Now add to each one information about where the actual reward is:
	States = [["p"+x+"_r"+y for x in BaseStates] for y in BaseStates]
	States = [item for sublist in States for item in sublist] + ['Death'] # Also add a death state
	
	# Create prior
	InitialBelief = [1.0/len(States)] * len(States)

	# Create actions: move to position x
	Actions = ["m"+x for x in BaseStates] + ['Take']
	
	# Now create transition matrix:
	# action, start-state, end-state, and probability.
	T = [[x, y, z, 0] for x in Actions for y in States for z in States]
	for index in range(len(T)):
		# First check if you're in death state:
		if T[index][1] == 'Death':
			if T[index][2] == 'Death':
				T[index][3] = 1
		# Now that we now it's not a death state.
		elif T[index][0] == 'Take':
			# Then only if the state is death
			if T[index][2] == 'Death':
				T[index][3] = 1
		else:
			# You're not in death state and you're not moving towards death state.
			IntendedState = T[index][0][1:] # This extracts the position of the movement
			FinalState = T[index][2][1:4] # This is the position of the state
			# To pass, both the states, and the reward position need to match.
			# Check states:
			if IntendedState == FinalState:
				# Check reard match:
				if T[index][1][-3:] == T[index][2][-3:]:
					T[index][3] = 1
	
	# Now create observation matrix:
	Observations = ['e','f'] # empty or full
	# action, end-state, observation and probability, respectively.
	O = [['*', s, o, 0] for s in States for o in Observations]
	for index in range(len(O)):
		if O[index][1] == 'Death':
			# Then just always give observation empty:
			if O[index][2] == 'e':
				O[index][3] = 1
		else:
			# If state where you end has the object in the position, then you get observation 1
			TargetState = O[index][1]
			if TargetState[1:4] == TargetState[6:]:
				# They match, so you get observation full
				if O[index][2] == 'f':
					O[index][3] = 1
			else:
				# They don't match, so you get observation empty
				if O[index][2] == 'e':
					O[index][3] = 1
	
	# Create costs and rewards:
	#  action, start-state, end-state, observation and the reward, respectively. 
	R = [[a, s, z, '*', 0] for a in Actions for s in States for z in States]
	for index in range(len(R)):
		ExecState = R[index][1]
		FinalState = R[index][2]
		# If you're taking a 'take' action then you get a reward if you got the right drawer
		if R[index][0] == 'Take':
			# Take action will always send you to Death state,
			# so this additional conditional is not necessary but reduces file size of .POMDP anad doesn't affect planning.
			if R[index][2] == 'Death':
				if ExecState[1:4] == ExecState[6:]:
					R[index][4] = PlanningReward
		else:
			# You did not take a take action, so set the cost to the distance,
			# but only if you're not between death states
			if ExecState != 'Death' and FinalState != "Death":
				# First check that action and final state align.
				# Otherwise we can add a cost, but it would just never happen so this helps make the .POMDP file shorter:
				if R[index][0][1:4] == R[index][2][1:4]:
					Startcoords = [int(x) for x in ExecState[1:4].split('-')]
					Endcoords = [int(x) for x in FinalState[1:4].split('-')]
					# Now compute the cost:
					Cost = np.sqrt((Startcoords[0]-Endcoords[0])**2+(Startcoords[1]-Endcoords[1])**2)
					R[index][4] = Cost*ActionCost
	return [TrialName, FutureDiscount, States, Actions, Observations, InitialBelief, T, O, R]

def PrintPOMDP(POMDPList):
	"""
	Takes the exact output of BuildPOMDP and creates a .POMDP file.
	"""
	[TrialName, FutureDiscount, States, Actions, Observations, InitialBelief, T, O, R] = POMDPList
	TrialName = TrialName.rstrip('\r\n')
	filewriter = open(TrialName+".POMDP","w+")
	filewriter.write('discount: '+str(FutureDiscount)+'\nvalues: reward\n')
	PrintList('states', States, filewriter)
	PrintList('actions', Actions, filewriter)
	PrintList('observations', Observations, filewriter)
	PrintList('start', InitialBelief, filewriter)
	# Print transition matrix:
	filewriter.write('T: * : * : * 0.000000\n')
	for tstn in T:
		if tstn[3]>0:
			filewriter.write('T: ' + tstn[0] + ' : ' + tstn[1] + ' : ' + tstn[2] + ' ' + str(tstn[3]) + '\n')
	# Print observation matrix:
	filewriter.write('O: * : * : * 0.000000\n')
	for obs in O:
		if obs[3]>0:
			filewriter.write('O: ' + obs[0] + ' : ' + obs[1] + ' : ' + obs[2] + ' ' + str(obs[3]) + '\n')
	# Print reward matrix:
	filewriter.write('R: * : * : * : * 0\n')
	for reward in R:
		# print(reward)
		if reward[4]!=0:
			filewriter.write('R: ' + reward[0] + ' : ' + reward[1] + ' : ' + reward[2] + ' : ' + reward[3] + ' ' + str(reward[4]) + '\n')
	filewriter.close()

DrawerDimensions = [3, 4] # 3 rows, 4 columns
TrialName = "Drawer3x4space"

sys.stdout.write("Generating "+TrialName+".POMDP... ")
PrintPOMDP(BuildPOMDP(TrialName,DrawerDimensions))
sys.stdout.write("Done!\n")
