import monitor
import torch
import math
import numpy as np

'''
class Params:
  def __init__(self, A: np.float64, b: np.float64, c: np.float64, d: np.float64, ep: np.float64, m: np.float64):
    self.A = A
    self.b = b
    self.c = c
    self.d = d
    self.ep = ep
    self.m = m
  def __str__(self) -> str:
    return "Params(" + "A=" + str(self.A) + ", " + "b=" + str(self.b) + ", " + "c=" + str(self.c) + ", " + "d=" + str(self.d) + ", " + "ep=" + str(self.ep) + ", " + "m=" + str(self.m) +  ")"

# State (control choices, environment measurements etc.)
class State:
  def __init__(self, a: np.float64, p: np.float64, t: np.float64, v: np.float64):
    self.a = a
    self.p = p
    self.t = t
    self.v = v
  def __str__(self) -> str:
    return "State(" + "a=" + str(self.a) + ", " + "p=" + str(self.p) + ", " + "t=" + str(self.t) + ", " + "v=" + str(self.v) +  ")"

'''
#Use boundaryDist function
# **pre, curr States are in regards to when controller changes acceleration? 

def potential_func(state, parameters): 
    """
    state and parameters are dictionaries with required args
    """

    # change 0 to np.float64(0.0)
    pre = monitor.State(
        a = state['prev_acceleration'],
        p = state['position'], 
        t = np.float64(0.0), 
        v = state['velocity']
    )

    curr = monitor.State(
        a = state['acceleration'],
        p = state['position'], 
        t = np.float64(0.0), 
        v = state['velocity']
    )
    # t = 0 from last monitor configuration 

    params = monitor.Params(
            A = parameters['acc'], 
            b = (-1 * parameters['braking']), 
            c = state['current_speed_limit'], 
            d = state['next_speed_limit'],
            ep = 0.1, 
            m = state['next_speed_limit_position'])
    #where the speed limit BEGINS 

    result = monitor.boundaryDist(pre, curr, params)

    return result 



def potential_reward(y, current, previous):
    """
    y = discount factor
    current = output of potential function for current state
    previous = output of potential function for previous state
    """
    return (y * current.val) - previous.val
