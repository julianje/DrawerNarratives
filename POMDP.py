import numpy as np
import copy
import math
from xml.dom import minidom
import re
import Hypothesis

class POMDP:

	def __init__(self, POMDPFile, PolicyFile):
		"""
		Both POMDPFile and PolicyFile are strings that indicate where to find the
		.POMDP object, and the .policy object trained in appl.

		policy is a list with two items, a matrix of alpha vectors and
		the list of corresponding actions
		"""
		self.POMDPFile = POMDPFile
		self.PolicyFile = PolicyFile
		self.policy = []
		self.states = []
		self.actions = []
		self.observations = []
		self.beliefs = []
		self.initialbeliefs = []
		self.T = []
		self.O = []

	def resetbeliefs(self):
		"""
		Helper function to reset beliefs and re-use pomdp for different observations
		"""
		self.beliefs = copy.deepcopy(self.initialbeliefs)

	def setbeliefs(self,beliefs):
		"""
		Push a set of beliefs to the agent:
		"""
		self.beliefs = beliefs

	def pAction(self, action, tau=0.1):
		"""
		Return probability of a specific action.
		Tau is kind of weird here just because the values are soo huge
		"""
		expvalues = np.matmul(self.policy[0], self.beliefs) # alphamatrix * beliefs
		expvalues = [max([expvalues[z] for z in [i for i, x in enumerate(self.policy[1]) if x == actionid]]) for actionid in range(len(self.actions))]
		maxval = abs(max(expvalues))
		expvalues = [x-maxval for x in expvalues]
		try:
			expvalues = [math.exp(x*1.0/tau) for x in expvalues]
			norm = sum(expvalues)
			if norm == 0:
				expvalues = [0 for x in expvalues]
			else:
				expvalues = [x/norm for x in expvalues]
			return expvalues[self.actions.index(action)]
		except OverflowError:
			print("Couldn't softmax!")
			raise

	def selectAction(self):
		"""
		Take a belief vector, an alphavectors matrix and a
		list of their corresponding actions and return best next action.
		"""
		expvalues = np.matmul(self.policy[0], self.beliefs) # alphamatrix * beliefs
		maxvalindex = np.where(expvalues == np.amax(expvalues))[0][0]
		# Get the action that corresponds to best alpha vector:
		actionId = self.policy[1][maxvalindex]
		return self.actions[actionId]

	# Create a function that updates beliefs.
	def updatebeliefs(self, action, observation):
		"""
		Updates the agent's beliefs, according to the action
		and observation they obtained.
		"""
		newbelief = [0] * len(self.beliefs)
		for so in range(len(self.beliefs)):
			for sf in range(len(self.beliefs)):
				newbelief[sf] += self.beliefs[so]*self.T[so,action,sf]*self.O[sf,action,observation]
		# Normalize
		if(sum(newbelief)>0):
			c = sum(newbelief)
			newbelief = [x*1.0/c for x in newbelief]
		self.beliefs = newbelief

	def GetNextState(self, state, action):
		"""
		Get next state, assuming a deterministic transition matrix
		"""	
		return np.where(self.T[state,action,:]==1)[0][0]
	
	def loadPOMDP(self):
		"""
		loads POMDP and optimal policy
		"""
		self.load_POMDP()
		self.load_policy()

	def load_policy(self):
		"""
		Load policy file
		"""
		# Load file and extract alpha-vectors
		mydoc = minidom.parse(self.PolicyFile)
		vectors = mydoc.getElementsByTagName('Vector')
		
		# Initialize matrix. Each row has an alpha vector.
		metadata = mydoc.getElementsByTagName('AlphaVector')
		#AlphaVector vectorLength="785" numObsValue="1" numVectors="62564"
		rowno = int(metadata[0].attributes['numVectors'].value)
		columnno = int(metadata[0].attributes['vectorLength'].value)
		alphamatrix = np.zeros((rowno,columnno))
		actions = [0] * rowno
		
		# Populate matrix and actions:
		for i in range(len(vectors)):
			# extract action (you need to do the parsing to str and then int)
			actions[i] = int(str(vectors[i].attributes['action'].value))
			# fill matrix
			rawdata = vectors[i].firstChild.data.split(' ')
			vecvals = [float(rawdata[j]) for j in range(len(rawdata)-1)]
			alphamatrix[i,:] = vecvals
		self.policy =[alphamatrix,actions]

	def load_POMDP(self):
		"""
		Take a .POMDP file and load into python
		"""
		f = open(self.POMDPFile, "r")
		disc_line = f.readline() # discount line
		discount = float(disc_line.strip('discount: ').strip('\n'))
		trash_line = f.readline() # values: rewards line, discard.
		states_line = f.readline() # States:
		states_line = states_line.strip('states: ').strip('\n')
		states = states_line.split(' ')
		actions_line = f.readline() # actions:
		actions_line = actions_line.strip('actions: ').strip('\n')
		actions = actions_line.split(' ')
		observations_line = f.readline() # observations
		observations_line = re.sub('observations: ','',observations_line)
		observations_line = observations_line.strip('\n')
		observations = observations_line.split(' ')
		beliefs_line = f.readline() # starting beliefs
		beliefs_line = beliefs_line.strip('start: ').strip('\n')
		beliefs = beliefs_line.split(' ')
		beliefs = [float(x) for x in beliefs]
		transition = f.readline() # This line just says to initialize T at 0.
		T = np.zeros((len(states),len(actions),len(states)))
		# Now read transition lines:
		inputline = f.readline()
		transitionprocess = True
		while(transitionprocess):
			if "T: " in inputline:
				[act,so,sf,val] = self.ProcessStateLine(inputline, actions, states)
				T[so,act,sf]=val
				inputline = f.readline()
			else:
				transitionprocess = False
		# when you leave the loop, you're at the O * * * 0 line
		O = np.zeros((len(states),len(actions),len(observations)))
		inputline = f.readline()
		observationprocess = True
		while(observationprocess):
			if "O: " in inputline:
				[so,sf,val] = self.ProcessObservationLine(inputline, actions, states, observations)
				O[so,:,sf] = val
				inputline = f.readline()
			else:
				observationprocess = False
		# Leaving the loop you're at the reward stage.
		# don't process it for now.
		self.states = states
		self.actions = actions
		self.observations = observations
		self.initialbeliefs = beliefs
		self.beliefs = beliefs
		self.T = T
		self.O = O

	def ProcessObservationLine(self,mystring, actions, states, observations):
		"""
		Take a string from an observation and return numbers you need
		"""
		mystring = re.sub('O: ','',mystring)
		mystring = mystring.strip("\n")
		components = mystring.split(' : ')
		initialstate = states.index(components[1])
		lastcomponent = components[2].split(' ')
		targetobs = observations.index(lastcomponent[0])
		modelval = float(lastcomponent[1])
		return([initialstate,targetobs,modelval])
	
	def ProcessStateLine(self,mystring, actions, states):
		"""
		Take a string from a transition and return numbers you need
		"""
		mystring = re.sub('T: ','',mystring)
		mystring = mystring.strip("\n")
		components = mystring.split(' : ')
		actionindex = actions.index(components[0])
		initialstate = states.index(components[1])
		lastcomponent = components[2].split(' ')
		targetstate = states.index(lastcomponent[0])
		modelval = float(lastcomponent[1])
		return([actionindex,initialstate,targetstate,modelval])








