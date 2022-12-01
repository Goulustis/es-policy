import numpy as np 

"""
ASSERT: ALL FUNCTIONS BE IN [-1, 1]
"""


def fix_shape(pos, ret_xy = False):
    if pos.shape[-1] != 2:
        pos = pos.T
    
    if ret_xy:
        return pos[...,0], pos[..., 1]
    return pos

def parabola(pos):
    pos = fix_shape(pos)

    return (1 - pos**2).sum(axis = -1)

def narrowing_path(pos):
    x, y = fix_shape(pos, ret_xy=True)
   
    vals = 8 - 2*((x - 1)**2 +(y)**2)
    bounds = 1/(15**(x+1.1))+0.05
    vals[y >= bounds] = 0
    vals[y <= -bounds] = 0
    return vals

def donut(pos):
    pos = fix_shape(pos)
    num = (4*pos**2).sum(axis=-1) # numerator

    to_sub = 5*(np.sqrt(num) < 0.05)

    return  1-(num) - to_sub 


def cont_grad(pos):
    x, y = fix_shape(pos, ret_xy=True)
    return 4 - (2*(x - 1))**2 - y**2

def grad_gap(pos):
    x, y = fix_shape(pos, ret_xy=True)
    res = cont_grad(pos)
    res[((x >= 0.05) & (x <= 0.1))] = -10
    return res 


def grad_cliff(pos):
    x, y = fix_shape(pos, ret_xy=True)
    res = cont_grad(pos)
    res[((x >= 0.05) )] = -13
    return res 


def fleeting_peaks(pos):
    x, y = fix_shape(pos, ret_xy=True)

    centers = [-0.75, -0.325, 0.05, 0.425]
    hs = [2.5, 4.5, 6, 7] # hights at respective centers
    r = 0.001
    
    res = 8-((3*y)**2+(x-2)**2)
    for c,h in zip(centers, hs):
        res[((x-c)**2 + y**2) <= r] = h

    return res 

def narrowing_peaks(pos):
    x, y = fix_shape(pos, ret_xy=True)

    centers = [-0.75, -0.325, 0.05, 0.425]
    hs = [2.5, 4.5, 6, 7] # hights at respective centers
    r = 0.0005
    # r = 0.005
    
    res = 8-((3*y)**2+(x-2)**2)
    for c,h in zip(centers, hs):
        res[((x-c)**2 + y**2) <= r] = h
    
    bounds = 1/(15**(x+1.1))+0.05
    res[y >= bounds] = -1
    res[y <= -bounds] = -1

    return res 

def narrowing_peaks_large(pos):
    x, y = fix_shape(pos, ret_xy=True)

    centers = [-0.75, -0.325, 0.05, 0.425]
    hs = [2.5, 4.5, 6, 7] # hights at respective centers
    r = 0.005
    
    res = 8-((3*y)**2+(x-2)**2)
    for c,h in zip(centers, hs):
        res[((x-c)**2 + y**2) <= r] = h
        
    
    bounds = 1/(15**(x+1.1))+0.05
    res[y >= bounds] = -1
    res[y <= -bounds] = -1

    return res 


def corner_peak(pos):
    x, y = fix_shape(pos, ret_xy=True)
    vals = 8 - 4*(0.2*(x - 1.5)**2 +(y)**2)
    bounds = 1/(15**(x+1.9))+0.05
    vals[y >= bounds] = 0
    vals[y <= -bounds] = 0
    vals[x >= 0.25] = 0

    para = 8 - 2*((x - .8)**2 +(y+0.8)**2)
    cond = (x >= 0.3) & (y < -0.1)
    vals[cond] = para[cond]
    return vals

FITNESS_FNS = {"parabola": {"fitness_fnc": parabola,
                            "start_pos" : [-0.95, -0.95]},
               "narrowing_path":{"fitness_fnc": narrowing_path,
                                "start_pos" : [-0.95,0]},
               "donut":{"fitness_fnc": donut,
                                "start_pos" : [-0.95,0]},
               "cont_grad":{"fitness_fnc":cont_grad,
                                "start_pos" : [-0.95, 0]},
               "grad_gap":{"fitness_fnc":grad_gap,
                                "start_pos" : [-0.95, 0]},
               "grad_cliff":{"fitness_fnc":grad_cliff,
                                "start_pos" : [-0.95, 0]},
               "fleeting_peaks":{"fitness_fnc":fleeting_peaks,
                                "start_pos" : [-0.95, 0]},
               "corner_peak":{"fitness_fnc":corner_peak,
                                "start_pos" : [-0.95, 0]},
               "narrowing_peaks":{"fitness_fnc":narrowing_peaks,
                                   "start_pos" : [-0.95, 0]},
               "narrowing_peaks_large":{"fitness_fnc":narrowing_peaks_large,
                                   "start_pos" : [-0.95, 0]}
                }