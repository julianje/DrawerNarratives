import numpy as np
import POMDP
import sys
import Hypothesis
from itertools import permutations
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar
from math import inf
import csv
import time

class Observer:

	def __init__(self, POMDPFile, PolicyFile, OpenDrawers, DrawerDimensions, ColorCoding = None):
		"""
		Take a .POMDP file, a .policy file, and a set of open drawers
		"""
		self.POMDPFile = POMDPFile
		self.PolicyFile = PolicyFile
		self.OpenDrawers = OpenDrawers
		self.DrawerDimensions = DrawerDimensions
		self.Model = POMDP.POMDP(POMDPFile, PolicyFile)
		self.KHypotheses = [] # Hypotheses about what the agent knows
		self.HandPosition = [] # Distribution of hand starting point
		self.Posterior = []
		self.RecallPrior = None # This is the probability that an agent will suddenly remember.
		if ColorCoding is None:
			self.DrawerColors = np.zeros(self.DrawerDimensions)
		else:
			self.DrawerColors = ColorCoding

	def load(self):
		self.Model.loadPOMDP()

	def ComputeActionDistance(self, actions, cultural=False):
		"""
		Internal supporting function.
		Take a list of actions and compute total distance that the hand travelled.
		When cultural is set to False it computes euclidean distance.
		When cultural is set to True we do account for cultural costs meaning:
		(i) moving from right-most drawer to left-most one in a consecutive way is a cost of 0
		(ii) oving horizontally is faster than moving vertically
		"""
		# Transform actions into coordinates
		ActionCoords = [[int(x[1]),int(x[3])] for x in actions]
		if not cultural:
			dist = sum([(ActionCoords[i-1][0]-ActionCoords[i][0])**2 + (ActionCoords[i-1][1]-ActionCoords[i][1])**2 for i in range(1,len(ActionCoords))])
		else:
			dist = 0
			for i in range(1,len(ActionCoords)):
				d = 1.5*(ActionCoords[i-1][0]-ActionCoords[i][0])**2 + (ActionCoords[i-1][1]-ActionCoords[i][1])**2
				# Check if you're not moving from end of a drawer to beginning of next.
				# First check is whether you're on the last column and whether the next state you're in the first row
				if (ActionCoords[i-1][1] == (self.DrawerDimensions[1]-1)) and ActionCoords[i][1] == 0:
					# Now check if you moved exactly one row down:
					if ActionCoords[i][0] == ActionCoords[i-1][0]+1:
						#sys.stdout.write(str(ActionCoords[i-1])+" // "+str(ActionCoords[i])+"\n")
						d = 0 # Delete distance
				dist += d
		return dist

	def InferKnowledge(self, tau=0.1, progressbar = True, dumboptimize=100, cultural=False):
		"""
		Infer what initial knowledge best explains open drawers
		This code first generates all hypotheses, and then duplicates each one to account for potential remembering
		This is for efficiency because if the agent remembered, it will always happen in the last frame.
		# Dumb optimize receives a percentage p and only considers the p% of shortest action paths.
		# The cultural flag is just piped into the distance computation since you need it for the dumboptimize
		"""
		self.BuildKHypothesisSpace() # Build knowledge hypothesis spaces
		self.InitializeHandPosition()
		# Re initialize posterior
		self.Posterior = []
		# now build space of actions:
		ActionSpace = [list(x) for x in permutations(self.OpenDrawers)]
		Distances = [self.ComputeActionDistance(x, cultural=cultural) for x in ActionSpace]
		sys.stdout.write("Original action space = "+str(len(ActionSpace))+".")
		# Now reduce action space based on dumboptimize percentage:
		ActionsConsidered = int(np.ceil(len(Distances)*dumboptimize/100))
		ActionIndices = np.argsort(Distances)[:ActionsConsidered]
		ActionSpace = [ActionSpace[x] for x in ActionIndices]
		sys.stdout.write("Revised action space = "+str(len(ActionSpace))+"\n")
		#for ac in ActionSpace:
		#	sys.stdout.write(str(ac)+"\n")
		# End of action space reduction
		DrawerPositions = [x[1:] for x in self.OpenDrawers] # This is just used to optimize
		if progressbar:
			bar = IncrementalBar('', max=len(self.KHypotheses)*len(ActionSpace), suffix='%(percent)d%%')
		for CurrHypothesis in self.KHypotheses:
			if CurrHypothesis[0] == "Play":
				# In play, assume costs did not factor into choice but agent was noenetheless efficient
				StateSpaceSize = self.DrawerDimensions[0] * self.DrawerDimensions[1]
				# Prior times likelihood
				p = np.log(CurrHypothesis[2]) + sum([np.log(1.0/(16-i)) for i in range(len(self.OpenDrawers))]) # Uniform over any actions.
				# Get most efficient action space:
				# This code is very similar to the one above, but adding the initial hand position to get the most efficient action space:
				ActionSpaceB = [['m0-0']+list(x) for x in permutations(self.OpenDrawers)]
				DistancesB = [self.ComputeActionDistance(x) for x in ActionSpaceB]
				Playactions = ActionSpace[np.argsort(DistancesB)[0]]
				myPosterior = Hypothesis.Hypothesis(CurrHypothesis[0],CurrHypothesis[1],self.HandPosition, Playactions, p)
				self.Posterior.append(myPosterior)
				continue
			# Ok now run inference for each combination once we know it's not a play hypothesis
			# QUICK OPTIMIZE: IF KNOWLEDGE HYPOTHESIS IS YOU KNOW IT'S IN X BUT YOU DIDN'T EVEN OPEN X
			# THEN PROBABILITY IS 0 SO JUST SKIP IT
			if CurrHypothesis[0][1:] not in DrawerPositions:
				continue # So we don't even store it. That way if it's not in the csv we know that it didn't get considered
			for actiontest in ActionSpace:
				if progressbar:
					bar.next()
				#sys.stdout.write('/')
				self.Model.setbeliefs(self.BuildInitialKnowledge(CurrHypothesis[1])) # Initialize knowledge
				p = np.log(CurrHypothesis[2]) # Initialize with the prior
				pmem = np.log(CurrHypothesis[2]) + np.log(self.RecallPrior)
				# assume all observations are empty except last one
				observations = ['e'] * len(actiontest)
				observations[-1] = 'f'
				# TO DUPLICATE EACH HYPOTHESES AND INCLUDE A KNOWLEDGE ONE
				for i in range(len(actiontest)-1):
					#sys.stdout.write('.')
					prob = self.Model.pAction(actiontest[i], tau=tau)
					Likelhood = np.log(prob) if prob != 0 else -inf
					p += Likelhood
					pmem += Likelhood
					#print(actiontest[i])
					#print(observations[i])
					actionid = self.Model.actions.index(actiontest[i])
					observationid = self.Model.observations.index(observations[i])
					self.Model.updatebeliefs(actionid, observationid)
				# Now there's just one final action
				# In the memory case, that last action has likelihood 1 since you'll go wherever you remember:
				# If we just do this, then the model considers hypotheses where you didn't know and you remembered before your first
				# action. That's technically correct, but is odd, so we'll add one conditional that sets memory to 0 if you remembered
				# before your first action!
				if len(self.OpenDrawers) == 1:
					Posterior = Hypothesis.Hypothesis('Mem-'+CurrHypothesis[0],CurrHypothesis[1],self.HandPosition, actiontest, np.log(0))
				else:
					Posterior = Hypothesis.Hypothesis('Mem-'+CurrHypothesis[0],CurrHypothesis[1],self.HandPosition, actiontest, pmem)
				self.Posterior.append(Posterior)
				# And now update to the last observation in the no memory case and store
				newprob = self.Model.pAction(actiontest[-1], tau=tau)
				p += np.log(newprob) if newprob != 0 else -inf
				myPosterior = Hypothesis.Hypothesis(CurrHypothesis[0],CurrHypothesis[1],self.HandPosition, actiontest, p)
				self.Posterior.append(myPosterior)
		if progressbar:
			bar.finish()

	def TranslateBelief(self, Belief):
		"""
		Helper function to print states from a probability distribution
		"""
		return [self.Model.states[x] for x in range(len(self.Model.states)) if Belief[x]>0]

	def BuildInitialKnowledge(self, Hypothesis):
		"""
		This function takes a hypothesis about where the reward is located and
		merges with beliefs about starting positions. Initialize by default to top-left drawer.
		"""
		IntegratedHypothesis = [Hypothesis[x]*self.HandPosition[x] for x in range(len(self.Model.states))]
		# Normalize
		IntegratedHypothesis = [x*1.0/sum(IntegratedHypothesis) for x in IntegratedHypothesis]
		return IntegratedHypothesis

	def InitializeHandPosition(self, InitialHandPosition = 'p0-0'):
		HandPosition = [0] * len(self.Model.states)
		for i in range(len(self.Model.states)):
			if self.Model.states[i][0:4] == InitialHandPosition:
				HandPosition[i] = 1
		self.HandPosition = HandPosition

	def BuildKHypothesisSpace(self):
		"""
		Build space of all hypotheses. Here hypotheses are states of knowledge.
		Just call different subroutines and then paste them all together.
		So each hypothesis is a probability distribution over world states.
		These only capture priors over reward positions, without considering
		knowledge about where the agent's hand begun.
		"""
		# IMPORTANT NOTE:
		# HYPOTHESES GENERATORS DO NOT RETURN PROPER PROBABILITY DISTRIBUTIONS
		# BUT THAT'S OKAY BECAUSE THE CONSTANT GETS ADJUSTED LATER.
		KnowledgeHypotheses = self.KnowledgeHypotheses(prior = 10)
		RowColumnHypotheses = self.LineHypotheses(prior = 2.5)
		MemoryHypotheses = self.MemoryHypotheses(prior = 2.5)
		ColorHypotheses = self.ColorHypotheses(prior = 5)
		IgnoranceHypothesis = self.IgnoranceHypothesis(prior = 15)
		PlayHypothesis = self.PlayHypothesis(prior = 1)
		# Create hypothesis with no knowledge:
		self.KHypotheses = KnowledgeHypotheses + RowColumnHypotheses + IgnoranceHypothesis + MemoryHypotheses + ColorHypotheses + PlayHypothesis
		# Now standardize the prior:
		Norm = sum([x[2] for x in self.KHypotheses])
		self.KHypotheses = [[x[0], x[1], x[2]/Norm] for x in self.KHypotheses]
		# Finally, add the probability of remembering. This is hypothesis independent, since it can happen with any hypothesis
		self.RecallPrior = 0.01

	def PlayHypothesis(self, prior=1):
		"""
		Build play hypothesis.
		THIS IS A PLACEHOLDER. When you're playing you don't care about the object
		The main function will just use a uniform, but we need some place holder for the main hypothesis space
		"""
		Play = [["Play", [1.0/len(self.Model.states)] * len(self.Model.states), prior]]
		return Play

	def IgnoranceHypothesis(self, prior=1):
		"""
		Build ignorance hypothesis
		"""
		IgnoranceHypothesis = [["Full ignorance", [1.0/len(self.Model.states)] * len(self.Model.states), prior]]
		return IgnoranceHypothesis

	def LineHypotheses(self, prior=1):
		"""
		Return all hypotheses where the agent knows either a row of a column
		"""
		# Build row hypotheses:
		RowHypotheses = [[None, [0] * len(self.Model.states), prior] for x in range(self.DrawerDimensions[0])]
		for i in range(self.DrawerDimensions[0]):
			RowHypotheses[i][0] = 'Row ' + str(i)
			for j in range(len(RowHypotheses[i][1])):
				# Here i corresponds to hypothesis that things are in row i.
				if self.Model.states[j][-3:-2] == str(i):
					RowHypotheses[i][1][j] = 1
		ColumnHypotheses = [[None, [0] * len(self.Model.states), prior] for x in range(self.DrawerDimensions[1])]
		for i in range(self.DrawerDimensions[1]):
			ColumnHypotheses[i][0] = 'Column ' + str(i)
			for j in range(len(ColumnHypotheses[i][1])):
				if self.Model.states[j][-1:] == str(i):
					ColumnHypotheses[i][1][j] = 1
		H = RowHypotheses + ColumnHypotheses
		return H

	def MemoryHypotheses(self, prior=1):
		"""
		Return hypotheses were agents has some vague sense of where the object is.
		"""
		# We'll do something similar to the knowledge one, but we'll add some graded noise a function of distance.
		# Reward locations:
		RewardPositions = [[x+0.5,y+0.5] for x in range(self.DrawerDimensions[0]-1) for y in range(self.DrawerDimensions[1]-1)]
		Hypotheses = [[None,[0] * len(self.Model.states), prior] for x in range(len(RewardPositions))]
		# Now create each hypothesis:
		for i in range(len(RewardPositions)):
			Reward = RewardPositions[i]
			Hypotheses[i][0] = 'Approx-' + '-'.join(map(str, Reward))
			MemoryCoords = Reward
			for j in range(len(Hypotheses[i][1])):
				if self.Model.states[j] != 'Death': # Death is not a position so that always has p=0
					# Now go through each state and add a probability as a function of the distance from the right drawer:
					statereward = [int(x) for x in self.Model.states[j][-3:].split('-')]
					# Now try something like the inverse of the distance:
					#sys.stdout.write("Memory: "+str(MemoryCoords)+", position"+str(statereward)+": ")
					# Option 1: distance to the 4th
					#denom = ((MemoryCoords[0]-statereward[0])**2 + (MemoryCoords[1]-statereward[1])**2)**2
					# Option 2: Exponential
					denom = 0.01**np.sqrt((MemoryCoords[0]-statereward[0])**2 + (MemoryCoords[1]-statereward[1])**2)
					if denom > 0:
						#sys.stdout.write(str(1.0/denom)+"\n")
						Hypotheses[i][1][j] = denom
					else:
						# sys.stdout.write("f2.0\n") # Not normalized, so just to make it twice as likely than one step away
						Hypotheses[i][1][j] = 1.0
		return Hypotheses

	def MemoryHypotheses_DrawerCenter(self, prior=1):
		"""
		Return hypotheses were agents has some vague sense of where the object is.
		This is old code that centers the memroy around a drawer, instead of at the corner of several drawers.
		"""
		# We'll do something similar to the knowledge one, but we'll add some graded noise a function of distance.
		# Reward locations:
		RewardPositions = list(set([self.Model.states[x].split('_')[1] for x in range(len(self.Model.states)-1)]))
		Hypotheses = [[None,[0] * len(self.Model.states), prior] for x in range(len(RewardPositions))]
		# Now create each hypothesis:
		for i in range(len(RewardPositions)):
			Reward = RewardPositions[i]
			Hypotheses[i][0] = 'Approx-' + Reward
			MemoryCoords = [int(x) for x in Reward[1:].split('-')]
			for j in range(len(Hypotheses[i][1])):
				if self.Model.states[j] != 'Death': # Death is not a position so that always has p=0
					# Now go through each state and add a probability as a function of the distance from the right drawer:
					statereward = [int(x) for x in self.Model.states[j][-3:].split('-')]
					# Now try something like the inverse of the distance:
					#sys.stdout.write("Memory: "+str(MemoryCoords)+", position"+str(statereward)+": ")
					# Option 1: distance to the 4th
					#denom = ((MemoryCoords[0]-statereward[0])**2 + (MemoryCoords[1]-statereward[1])**2)**2
					# Option 2: Exponential
					denom = 0.01**np.sqrt((MemoryCoords[0]-statereward[0])**2 + (MemoryCoords[1]-statereward[1])**2)
					if denom > 0:
						#sys.stdout.write(str(1.0/denom)+"\n")
						Hypotheses[i][1][j] = denom
					else:
						# sys.stdout.write("f2.0\n") # Not normalized, so just to make it twice as likely than one step away
						Hypotheses[i][1][j] = 1.0
		return Hypotheses

	def ColorHypotheses(self, prior=1):
		"""
		Return all hypotheses where the agent knows it's in one of the colors.
		"""
		if np.array_equal(self.DrawerColors, np.zeros(self.DrawerDimensions)):
			return []
		colors = np.unique(self.DrawerColors)
		Hypotheses = [[None,[0] * len(self.Model.states), prior] for x in range(len(colors))]
		for colorindex in range(len(colors)):
			indices = np.where(self.DrawerColors == colors[colorindex]) # Retrieve color positions
			Hypotheses[colorindex][0] = 'color_' + str(colors[colorindex])
			for position in range(len(indices[0])): # 0 or 1 are fine, since this is two arrays, one with x and one with y coordiantes.
				xval = indices[0][position]
				yval = indices[1][position]
				# Now cycle through state space and mark the relevant ones
				for stateindex in range(len(self.Model.states)-1): # Except Death state
					# Retrieve position of reward and check if it matches:
					if xval==int(self.Model.states[stateindex][-3:-2]) and yval==int(self.Model.states[stateindex][-1:]):
						Hypotheses[colorindex][1][stateindex] = 1
		return Hypotheses

	def KnowledgeHypotheses(self, prior=1):
		"""
		Return all hypotheses where the agent already knows where the object is,
		for each possible object position
		"""
		# Reward locations:
		RewardPositions = list(set([self.Model.states[x].split('_')[1] for x in range(len(self.Model.states)-1)]))
		Hypotheses = [[None,[0.01] * len(self.Model.states), prior] for x in range(len(RewardPositions))]
		# Now create each hypothesis:
		for i in range(len(RewardPositions)):
			Reward = RewardPositions[i]
			Hypotheses[i][0] = Reward
			for j in range(len(Hypotheses[i][1])):
				# Now go through each state and check if it contains reward in the right place:
				statereward = self.Model.states[j][-4:]
				if statereward == Reward:
					Hypotheses[i][1][j] = 5
		return Hypotheses

	def PrintPosterior(self):
		"""
		Prints all non-zero hypotheses, ranked by probability
		"""
		# First retrieve only hypotheses with p>0
		Probable = [x for x in self.Posterior if x.Belief>0]
		# Get beliefs
		Beliefs = [x.Belief for x in Probable]
		BeliefOrder = np.argsort(Beliefs)
		for i in range(len(BeliefOrder)-1,-1,-1): # Decreasing
			sys.stdout.write(Probable[BeliefOrder[i]].KnowledgeType+"\t"+str(Probable[BeliefOrder[i]].Actions)+"\t"+str(Probable[BeliefOrder[i]].Belief)+"\n") 

	def ProcessPosterior(self, rounded=True):
		"""
		The posterior is in this messier thing, so this function computes the marginals over actions and knowledge
		"""
		self.NormalizePosterior(rounded)
		# Now, marginalize over different things:
		ActionPosterior = {}
		KnowledgeTypePosterior = {}
		for i in range(len(self.Posterior)):
			Actions = '_'.join(self.Posterior[i].Actions)
			if Actions in ActionPosterior.keys():
				ActionPosterior[Actions] += self.Posterior[i].Belief
			else:
				ActionPosterior[Actions] = self.Posterior[i].Belief
			KT = self.Posterior[i].KnowledgeType
			if KT in KnowledgeTypePosterior.keys():
				KnowledgeTypePosterior[KT] += self.Posterior[i].Belief
			else:
				KnowledgeTypePosterior[KT] = self.Posterior[i].Belief
		return [ActionPosterior, KnowledgeTypePosterior]

	def NormalizePosterior(self, rounded=True):
		"""
		Integreate prior and normalize the posterior distrbution
		"""
		# First switch back to regular probabilities
		Norm = sum([np.exp(x.Belief) for x in self.Posterior])
		for i in range(len(self.Posterior)):
			self.Posterior[i].Belief = np.exp(self.Posterior[i].Belief)/Norm
			if rounded:
				self.Posterior[i].Belief = np.round(self.Posterior[i].Belief,2)

	def SaveResults(self,filename, tid=None, header=True):
		"""
		Save posterior.
		tid is TrialId
		"""
		Time = time.asctime(time.localtime(time.time()))
		with open(filename, 'a', newline='') as csvfile:
			filewriter = csv.writer(csvfile, delimiter=',')
			if header:
				filewriter.writerow(['TrialId','Timestamp','Drawers','Actions','Knowledge','Probability'])
			for H in self.Posterior:
				Actions = '_'.join(H.Actions)
				KT = H.KnowledgeType
				p = H.Belief
				Drawers = '_'.join(self.OpenDrawers)
				filewriter.writerow([tid, Time, Drawers, Actions, KT, p])

	def PlotPosterior(self, FinalPosterior, Title='', DeleteZeros=True):
		""" 
		Take a dictionary and plot it
		"""
		Knowledge = FinalPosterior[1]
		if DeleteZeros:
			Knowledge = {key:val for key, val in Knowledge.items() if val != 0}
		xvals = list(Knowledge.keys())
		yvals = list(Knowledge.values())
		plt.subplot(2, 1, 1)
		plt.title(Title)
		plt.barh(range(len(yvals)), yvals, align='center')
		plt.yticks(range(len(yvals)), xvals)
		plt.ylabel('Inferred knowledge')
		Actions = FinalPosterior[0]
		if DeleteZeros:
			Actions = {key:val for key, val in Actions.items() if val != 0}
		xvalsb = list(Actions.keys())
		yvalsb = list(Actions.values())
		plt.subplot(2, 1, 2)
		plt.barh(range(len(yvalsb)), yvalsb, align='center')
		plt.yticks(range(len(yvals)), xvalsb)
		plt.ylabel('Inferred actions')
		plt.show()

