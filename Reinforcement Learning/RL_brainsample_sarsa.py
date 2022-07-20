import numpy as np
import pandas as pd

DEBUG=1

def debug(debuglevel, msg, **kwargs):
    if debuglevel <= DEBUG:
        if 'printNow' in kwargs:
            if kwargs['printNow']:
                print(msg)
        else:
            print(msg)

class rlalgorithm:
    '''States are dynamically added to datastructure'''

    def check_state_exist(self, state):
        debug(3, '(checking state...)')
        if state not in self.Q:
            self.Q[state] = np.zeros(self.num_actions)
            debug(2, 'Adding state {}'.format(state))

    def __init__(self, actions, *args, **kwargs):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.alpha = 0.01
        self.display_name = "SARSA"
        self.Q = {}
        self.actions = actions
        self.num_actions = len(actions)
        debug(1, 'Init new RL Algorithm Basic: |A|={} A={} gamma={}'.format(self.num_actions, self.actions, self.gamma))

        pass

    def choose_action(self, observation):
        debug(3, '  (choosing action...)')
        self.check_state_exist(observation)
        debug(2, 'pi({})'.format(observation))
        debug(2, 'Q({})={}'.format(observation, self.Q[observation]))
        # Exploit
        if np.random.uniform() >= self.epsilon:
            a = self.actions[np.argmax(self.Q[observation])]
            debug(2, '   a_max: {}'.format(a))
        # Explore
        else:
            a = np.random.choice(self.actions)
            debug(2, '   a_rand: {}'.format(a))
        return a

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)

        # When a state has value 'terminal' is has no outgoing state, the game ends
        if s_ == 'terminal':
            a_ = np.random.choice(self.actions)
            self.Q[s][a] += self.alpha * self.gamma * r

        else:
            a_ = self.choose_action(s_)
            self.Q[s][a] += self.alpha * (r + self.gamma*self.Q[s_][a_]-self.Q[s][a])

        return s_, a_


