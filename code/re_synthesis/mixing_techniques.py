# -*- coding: utf-8 -*-
import numpy as np
from operator import add,mul
from const import SR, TEMPO

def mix(dj_set, new_track, style = 'basic'):
    
    if (style == 'basic'):
        dj_set = dj_set + new_track
        
    elif (style == 'noise'):
        length = int(1.2*SR)
        
        ramp = [float(i)/(length*3) for i in range(length)]
        tail = map(mul,np.random.rand(length),ramp)
        print(len(dj_set[-length:]), len(tail))
        dj_set[-length:] = map(add,dj_set[-length:],tail)
        dj_set = dj_set + new_track
        
    return dj_set