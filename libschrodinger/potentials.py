from enum import Enum
from dataclasses import dataclass
import numpy as np
import scipy.sparse as spsparse
import cupy as cp
import cupyx.scipy.sparse as cpsparse
import matplotlib.pyplot as plt
from typing import List
from typing import Tuple
import scipy.sparse.linalg as spla
import cupyx.scipy.sparse.linalg as cpla
from datetime import timedelta
from time import monotonic
import sys

def tunnelCase(position, where, width, potential = 1): 
    return np.where(
            (position.x > where) & (position.x < (where + width)), 
            potential, 
            0, 
        )

def hydrogenAtom(position, potential, bottom = 1): 
    return potential / np.sqrt(
            (position.x / 2) ** 2 \
            + (position.y / 2) ** 2 \
            + bottom ** 2 \
        )

def doubleSlit(position, where, width, slitHeight, gapHeight, potential = 1, math=np): 
    totalY = math.max(position.y)
    return math.where(
            (position.x > where) & (position.x < (where + width)) 
                    & ( \
                            (position.y > ((totalY / 2) + (gapHeight + slitHeight))) \
                            | (position.y < ((totalY / 2) - (gapHeight + slitHeight))) \
                            | ( \
                               (position.y > ((totalY / 2) - gapHeight)) \
                               & (position.y < ((totalY / 2) + gapHeight)) \
                              )
                      ), 
            potential, 
            0, 
        )

def constantPotentials(position, lengthRatios, potentialRatios, basePotential, math = np, epsilon = 1e-16): 
    regionCount = len(lengthRatios)
    assert regionCount == len(potentialRatios)
    assert (math.sum(lengthRatios) - 1.0) < epsilon
    potential = position.x * 0.0
    basePosition = math.min(position.x)
    xExtent = math.abs(math.max(position.x) - basePosition)
    for ii in range(regionCount): 
        regionEnd = basePosition + (xExtent * lengthRatios[ii])
        potential = math.where(
                (position.x >= basePosition) & (position.x < regionEnd), 
                potentialRatios[ii] * basePotential, 
                potential
            )
        basePosition = regionEnd
    return potential

def uniformTimeScaledPotentials(position, time, totalTime, delay, duration, lengthRatios, potentialRatios, potentialHeight, math):
    delay *= totalTime
    duration *= totalTime
    timeScalar = 0.0 if time < delay and time < (delay + duration) else math.sin(((time - delay) / duration) * np.pi)
    return constantPotentials(
            position, 
            lengthRatios, 
            potentialRatios, 
            potentialHeight * timeScalar, 
            math = math
        )
