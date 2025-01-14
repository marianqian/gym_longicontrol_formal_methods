#
# NFM23/Car Speed Limit.py
# Generated by KeYmaera X
# called monitor.py in mqian; edited 3/15/23

from typing import Callable
import numpy as np

# Model parameters
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

# Values for resolving non-deterministic assignments in control code
class Input:
  def __init__(self, a: np.float64):
    self.a = a
  def __str__(self) -> str:
    return "Input(" + "a=" + str(self.a) +  ")"

# Verdict identifier and value
class Verdict:
  def __init__(self, id: np.float64, val: np.float64):
    self.id = id
    self.val = val
  def __str__(self) -> str:
    return "Verdict(" + "id=" + str(self.id) + ", " + "val=" + str(self.val) +  ")"

def checkInit(init: State, params: Params) -> Verdict:
  (np.float64(0.0) <= init.v) and ((init.v <= params.c) and ((np.float64(0.0) <= params.d) and (((params.m)-(init.p) >= max((((init.v)*(init.v))-((params.d)*(params.d)))/((np.float64(2.0))*(params.b)), np.float64(0.0))) and ((params.b > np.float64(0.0)) and ((params.A >= np.float64(0.0)) and (params.ep >= np.float64(0.0)))))))

def Cond_1822348951(pre, curr, params):
  return (params.m)-(pre.p) >= ((((pre.v)+(max((pre.v)+((curr.a)*(params.ep)), np.float64(0.0))))*(params.ep))/(np.float64(2.0)))+(max((((max((pre.v)+((curr.a)*(params.ep)), np.float64(0.0)))*(max((pre.v)+((curr.a)*(params.ep)), np.float64(0.0))))-((params.d)*(params.d)))/((np.float64(2.0))*(params.b)), np.float64(0.0)))
def Margin_1822348951(pre, curr, params):
  return ((params.m)-(pre.p))-(((((pre.v)+(max((pre.v)+((curr.a)*(params.ep)), np.float64(0.0))))*(params.ep))/(np.float64(2.0)))+(max((((max((pre.v)+((curr.a)*(params.ep)), np.float64(0.0)))*(max((pre.v)+((curr.a)*(params.ep)), np.float64(0.0))))-((params.d)*(params.d)))/((np.float64(2.0))*(params.b)), np.float64(0.0))))
def Margin_2041402958(pre, curr, params):
  return (params.A)-(curr.a)
def Margin1664663636(pre, curr, params):
  return params.ep
def Margin79382589(pre, curr, params):
  return (params.c)-((pre.v)+((curr.a)*(params.ep)))
def Margin_1205389256(pre, curr, params):
  return pre.v
def Margin_79975035(pre, curr, params):
  return max(curr.t, -(curr.t))
def Margin_1902204806(pre, curr, params):
  return max((curr.v)-(pre.v), (pre.v)-(curr.v))
def Margin_375454166(pre, curr, params):
  return max((curr.p)-(pre.p), (pre.p)-(curr.p))
def Margin_968464967(pre, curr, params):
  return (-(params.b))-(curr.a)
def Or_1869392006(pre, curr, params):
  if curr.a <= params.A:
    if (pre.v)+((curr.a)*(params.ep)) <= params.c:
      if Cond_1822348951(pre, curr, params):
        if pre.v >= np.float64(0.0):
          if np.float64(0.0) <= params.ep:
            if curr.p == pre.p:
              if curr.v == pre.v:
                if curr.t == np.float64(0.0):
                  verdicts = [
                   Verdict(1, Margin_1205389256(pre, curr, params)),
                   Verdict(1, Margin_2041402958(pre, curr, params)),
                   Verdict(1, Margin79382589(pre, curr, params)),
                   Verdict(1, Margin1664663636(pre, curr, params)),
                   Verdict(1, Margin_1822348951(pre, curr, params))
                   
                  ]
                  nonMeasureZeroVerdicts = filter(lambda v: v.id != 0, verdicts)
                  return min(nonMeasureZeroVerdicts, key=lambda v: v.val, default=Verdict(0, True))
                else:
                  return Verdict(-1, -Margin_79975035(pre, curr, params))
              else:
                return Verdict(-2, -Margin_1902204806(pre, curr, params))
            else:
              return Verdict(-3, -Margin_375454166(pre, curr, params))
          else:
            return Verdict(-4, Margin1664663636(pre, curr, params))
        else:
          return Verdict(-5, Margin_1205389256(pre, curr, params))
      else:
        return Verdict(-6, Margin_1822348951(pre, curr, params))
    else:
      return Verdict(-7, Margin79382589(pre, curr, params))
  else:
    return Verdict(-8, Margin_2041402958(pre, curr, params))
def Or1523633005(pre, curr, params):
  if curr.a <= -(params.b):
    if pre.v >= np.float64(0.0):
      if np.float64(0.0) <= params.ep:
        if curr.p == pre.p:
          if curr.v == pre.v:
            if curr.t == np.float64(0.0):
              verdicts = [
               Verdict(1, Margin1664663636(pre, curr, params)),
               Verdict(1, Margin_1205389256(pre, curr, params)),
               Verdict(1, Margin_968464967(pre, curr, params))
               
              ]
              nonMeasureZeroVerdicts = filter(lambda v: v.id != 0, verdicts)
              return min(nonMeasureZeroVerdicts, key=lambda v: v.val, default=Verdict(0, True))
            else:
              return Verdict(-1, -Margin_79975035(pre, curr, params))
          else:
            return Verdict(-2, -Margin_1902204806(pre, curr, params))
        else:
          return Verdict(-3, -Margin_375454166(pre, curr, params))
      else:
        return Verdict(-4, Margin1664663636(pre, curr, params))
    else:
      return Verdict(-5, Margin_1205389256(pre, curr, params))
  else:
    return Verdict(-9, Margin_968464967(pre, curr, params))

def boundaryDist(pre: State, curr: State, params: Params) -> Verdict:
  '''
  Computes distance to safety boundary on prior and current state (>=0 is safe, <0 is unsafe)
  '''
  verdicts = [
   
   Or_1869392006(pre, curr, params),
   Or1523633005(pre, curr, params)
  ]
  nonMeasureZeroVerdicts = filter(lambda v: v.id != 0, verdicts)
  return max(nonMeasureZeroVerdicts, key=lambda v: v.val, default=Verdict(0, True))

def monitorSatisfied(pre: State, curr: State, params: Params) -> bool:
  '''
  Evaluates monitor condition in prior and current state
  '''
  return boundaryDist(pre,curr,params).val >= 0

def monitoredCtrl(curr: State, params: Params, inp: State,
                  ctrl: Callable[[State, State, Params], State],
                  fallback: Callable[[State, State, Params], State]) -> State:
  '''
  Run controller `ctrl` monitored, return `fallback` if `ctrl` violates monitor
  '''
  pre = curr
  post = ctrl(pre,params,inp)
  if monitorSatisfied(pre,post,params) == True:
    return post
  else:
    return fallback(pre,params,inp)
