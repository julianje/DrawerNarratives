import numpy as np
import POMDP
import sys
import Hypothesis
from itertools import permutations
import matplotlib.pyplot as plt

class Observer:

	def __init__(self, POMDPFile, PolicyFile, OpenDrawers, DrawerDimensions):
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

	def load(self):
		self.Model.loadPOMDP()

	def InferKnowledge(self):
		"""
		Infer what initial knowledge best explains open drawers
		"""
		self.BuildKHypothesisSpace() # Build knowledge hypothesis spaces
		self.InitializeHandPosition()
		for CurrHypothesis in self.KHypotheses:
			# now construct space of actions:
			ActionSpace = [list(x) for x in permutations(self.OpenDrawers)]
			# Ok now run inference for each combination:
			for actiontest in ActionSpace:
				self.Model.setbeliefs(self.BuildInitialKnowledge(CurrHypothesis[1])) # Initialize knowledge
				p = 0
				# assume all observations are empty except last one
				observations = ['e'] * len(actiontest)
				observations[-1] = 'f'
				for i in range(len(actiontest)):
					p += np.log(self.Model.pAction(actiontest[i]))
					print(actiontest[i])
					print(observations[i])
					actionid = self.Model.actions.index(actiontest[i])
					observationid = self.Model.observations.index(observations[i])
					self.Model.updatebeliefs(actionid, observationid)
				Posterior = Hypothesis.Hypothesis(CurrHypothesis[0],CurrHypothesis[1],self.HandPosition, actiontest, p)
				self.Posterior.append(Posterior)

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
		KnowledgeHypotheses = [["Perfect knowledge", x] for x in self.KnowledgeHypotheses()]
		RowColumnHypotheses = [["Line knowledge", x] for x in self.LineHypotheses()]
		# Create hypothesis with no knowledge:
		IgnoranceHypothesis = [["Full ignorance", [1.0/len(self.Model.states)] * len(self.Model.states)]]
		self.KHypotheses = KnowledgeHypotheses + RowColumnHypotheses + IgnoranceHypothesis

	def LineHypotheses(self):
		"""
		Return all hypotheses where the agent knows either a row of a column
		"""
		# Build row hypotheses:
		RowHypotheses = [[0] * len(self.Model.states) for x in range(self.DrawerDimensions[0])]
		for i in range(self.DrawerDimensions[0]):
			for j in range(len(RowHypotheses[i])):
				# Here i corresponds to hypothesis that things are in row i.
				if self.Model.states[j][-3:-2] == str(i):
					RowHypotheses[i][j] = 1
		ColumnHypotheses = [[0] * len(self.Model.states) for x in range(self.DrawerDimensions[1])]
		for i in range(self.DrawerDimensions[1]):
			for j in range(len(ColumnHypotheses[i])):
				if self.Model.states[j][-1:] == str(i):
					ColumnHypotheses[i][j] = 1
		H = RowHypotheses + ColumnHypotheses
		return H

	def KnowledgeHypotheses(self):
		"""
		Return all hypotheses where the agent already knows where the object is,
		for each possible object position
		"""
		# Reward locations:
		RewardPositions = list(set([self.Model.states[x].split('_')[1] for x in range(len(self.Model.states)-1)]))
		Hypotheses = [[0] * len(self.Model.states) for x in range(len(RewardPositions))]
		# Now create each hypothesis:
		for i in range(len(RewardPositions)):
			Reward = RewardPositions[i]
			for j in range(len(Hypotheses[i])):
				# Now go through each state and check if it contains reward in the right place:
				statereward = self.Model.states[j][-4:]
				if statereward == Reward:
					Hypotheses[i][j] = 1
		return Hypotheses

	def ProcessPosterior(self):
		"""
		The posterior is in this messier thing, so this function integrates things
		"""
		# First switch back to regular probabilities
		Norm = sum([np.exp(x.Belief) for x in self.Posterior])
		for i in range(len(self.Posterior)):
			self.Posterior[i].Belief = np.exp(self.Posterior[i].Belief)/Norm
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
		

	def PlotPosterior(self, FinalPosterior):
		""" 
		Take a dictionary and plot it
		"""
		plt.barh(range(len(FinalPosterior[0])), FinalPosterior[0], align='center')
		plt.yticks(range(len(FinalPosterior[0])), FinalPosterior[1])
		plt.show()

