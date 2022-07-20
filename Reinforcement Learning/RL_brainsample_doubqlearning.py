import numpy as np
DEBUG=1

def debug(debuglevel, msg, **kwargs):
    if debuglevel <= DEBUG:
        if 'printNow' in kwargs:
            if kwargs['printNow']:
                print(msg) 
        else:
            print(msg) 

class rlalgorithm:
                
    def __init__(self, actions, *args, **kwargs):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.alpha = 0.8
        self.display_name="Double QLearning"
        self.Q1={}
        self.Q2={}
        self.actions=actions
        self.num_actions = len(actions)
        debug(1, 'Init new RL Algorithm Basic: |A|={} A={} gamma={}'.format(self.num_actions, self.actions, self.gamma))

    def choose_action(self, observation):
        debug(3, '  (choosing action...)')
        self.check_state_exist(observation)
        debug(2, 'pi({})'.format(observation))
        if np.random.uniform() >= self.epsilon:
            a = self.actions[np.argmax(self.Q1[observation] + self.Q2[observation])]
        else:
            a = np.random.choice(self.actions)
        return a
            
    def learn(self, s, a, r, s_):
        debug(3, '  (learning...)')
        debug(2, 'Learn: s={}\n  a={}\n  r={}\n  s_={}'.format(s,a,r,s_))
        self.check_state_exist(s_)

        # When a state has value 'terminal' is has no outgoing state, the game ends
        if s_ == 'terminal':
            a = np.random.choice(self.actions)
            # why do this?
            self.Q[s][a] += self.alpha * self.gamma * r
        else:
            if np.random.uniform() < 0.5:
                self.Q1[s][a] += self.alpha * (r + self.gamma *  self.Q2[s_][np.argmax(self.Q1[s_])] - self.Q1[s][a])
            else:
                self.Q2[s][a] += self.alpha * (r + self.gamma *  self.Q1[s_][np.argmax(self.Q2[s_])] - self.Q2[s][a])
            a = self.choose_action(s)
        return s_, a


    '''States are dynamically added to datastructure'''
    def check_state_exist(self, state):
        debug(3, '(checking state...)')
        if state not in self.Q1:
            self.Q1[state] = np.zeros(self.num_actions)
            debug(2, 'Adding state {}'.format(state))
        if state not in self.Q2:
            self.Q2[state] = np.zeros(self.num_actions)
            debug(2, 'Adding state {}'.format(state))
 