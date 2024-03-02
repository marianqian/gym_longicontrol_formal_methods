import math
'''
def acc(val): 
    if val >= 0: return 0.05 * math.pow(val, 2) 
    else: return math.pow(val, 3)

def brake(val): 
    if val >= 0: return 10 + math.pow(val, 2) 
    else: return math.pow(val, 3)

def velocity(val):
    if val > 0: return 1 / val
    else: return math.pow(val, 3)


'''
def acc(val): 
    return val

def brake(val): 
    return val

def velocity(val): 
    return val 

def velocity2(val): 
    return val

'''
def velocity(val): 
    if val > 0: return val #only scaling negative aspects 
    elif val > -1: return -math.pow(-val, 1/3) 
    else: return math.pow(val, 3)
    #want to make it more negative since we don't want this behavior 
    # when pre.v <= pre.vdes is false, so pre.v > pre.vdes --> val = pre.vdes - pre.v 

def velocity2(val):  
    if val > 0: return val #only scaling negative aspects 
    elif val > -1: return math.pow(val, 3) 
    else: return -math.pow(-val, 1/3)
    #want to make it less negative since this behavior is ok 
    #when pre.v >= pre.vdes is false, so pre.v < pre.vdes --> val = pre.v - pre.vdes 
'''
'''
def velocity(val):
    return val
'''
