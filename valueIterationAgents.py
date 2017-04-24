# valueIterationAgents.py
# -----------------------
##
import mdp, util
import sys

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp = None, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbabilities(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        if (self.mdp != None):
            self.doValueIteration()

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        #"*** YOUR CODE STARTS HERE ***"
        nextStateProb = self.mdp.getTransitionStatesAndProbabilities(state, action)
        qvalue = 0
        for (sPrime, prob) in nextStateProb:
          qvalue += prob * (self.mdp.getReward(state,action,sPrime) + (self.discount * self.values[sPrime]))

        print "Q value", qvalue

        #util.raiseNotDefined()

        """
          This function is later used in doValueIteration and computeActionFromValues
          So we declare it beforehand
        """


        #"*** YOUR CODE FINISHES HERE ***"
        
        return qvalue
    

    def doValueIteration (self):
        # Write value iteration code here

        print "Iterations: ", self.iterations
        print "Discount: ", self.discount
        states = self.mdp.getStates()
        maxDelta = float("-inf")

        #"*** YOUR CODE STARTS HERE ***"
        # Your code should include the implementation of value iteration
        # At the end it should show in the terminal the number of states considered in self.values and
        # the Delta between the last two iterations
        
        # set all initial V(s) to zero 
        v1 = dict([(state,0) for state in states])
        # iterate number of times defined by program
        for i in range(0,self.iterations - 1):
          # get copy of previous values so that you can calculate delta
          v = v1.copy()
          delta = 0
          for s in states:
            # for each state compute eqn for each action
            vDict = dict([(action,0) for action in self.mdp.getPossibleActions(s)])
            if len(vDict) != 0:
              for a in vDict:
                # get probabilities for next states based on given state, and action
                for (sPrime, prob) in self.mdp.getTransitionStatesAndProbabilities(s,a):
                  vDict[a] += prob * (self.mdp.getReward(s,a,sPrime) + v[sPrime])
                  print 'vDict: ', a, ' = ', vDict[a]  
              print 'vDict: ', vDict
              v1[s] = vDict[max(vDict)]
            else :
              v1[s] = 0
          delta = max(delta, abs(v1[s] - v[s]))

        print 'Values in States:', v 
        print 'Delta: ', delta
      self.mdp.values = v
           # v[s'] = max(cost(action) + P(s|s') + P())
        #util.raiseNotDefined()
        #"*** YOUR CODE FINISHES HERE ***"
        
    def setMdp( self, mdp):
        """
          Set an mdp.
        """
        self.mdp = mdp
        self.doValueIteration()

    def setDiscount( self, discount):
        """
          Set a discount
        """
        self.discount = discount

    def setIterations( self, iterations):
        """
          Set a number of iterations
        """
        self.iterations = iterations
       
       
    def getValue(self, state):
        """
          Return the value of the state
        """
        return self.values[state]
        


    def showPolicy( self ):

        """
          Print the policy
        """
        
        states = self.mdp.getStates()
        for state in states:
            print "Policy\n", state, self.getPolicy(state)


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        #"*** YOUR CODE STARTS HERE ***"
        possibleActions = self.mdp.getPossibleActions(state)
        
        #If no legal actions return none
        if len(possibleActions) == 0:
          return None 

        qDict = {}
        for i in possibleActions:
          qDict[i] = self.computeQValueFromValues(state,i)
        
        # will return first element found if there is a tie
        bestAction = max(qDict)

        print "Best Action ", bestAction

        return bestAction
        #util.raiseNotDefined()

        #"*** YOUR CODE FINISHES HERE ***"

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getPolicy(self, state):
        "Returns the policy at the state (no exploration)."
        return self.getAction(state)

    
    def getQValue(self, state, action):
        "Returns the Q value."        
        return self.computeQValueFromValues(state, action)

    def getPartialPolicy(self, stateL):
        "Returns the partial policy at the state. Random for unkown states"        
        state,state_names = self.mdp.stateToHigh(stateL)
        if self.mdp.isKnownState(state):
            return self.computeActionFromValues(state)
        else:
            # random action
            return util.random.choice(stateL.getLegalActions()) 

