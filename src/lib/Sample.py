from __future__ import absolute_import, print_function

from collections import namedtuple

""" Named tuple: Sample 
    Composed of:
    state: current state
    action: current action
    reward: reward obtained by taking action in state and transitioning to next_state
    next_state: next state
    """
Sample = namedtuple('Sample', ('state', 'action', 'reward', 'next_state'))

def toStr(sample):
    if not isinstance(sample, Sample):
        return sample
    vals = [sample.state, sample.action, sample.reward, sample.next_state]
    str_vals = [str(v) for v in vals]
    key_vals = ['state', 'action', 'reward', 'next_state']
    ret_str = ['(%s=%s)'%(k,v) for k,v in zip(key_vals, str_vals)]
    return '],['.join(ret_str)
