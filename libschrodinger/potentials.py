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

@dataclass
class Rectangle2D: 
    x : float
    y : float
    width : float
    height : float

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

def constantPotentials(
            position, 
            lengthRatios, 
            potentialRatios, 
            basePotential, 
            math = np, 
            epsilon = 1e-16
        ): 
    regionCount = len(lengthRatios)
    assert regionCount == len(potentialRatios)
    potentialRatios = math.array(potentialRatios)
    lengthRatios = math.array(lengthRatios)
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

def axisAlignedBlocks(
            position, 
            boxes : List[Rectangle2D], 
            potentialRatios : List[float], 
            potentialHeight : float, 
            math = np
        ): 
    assert len(potentialRatios) == len(boxes)
    potential = position.x * 0.0
    #print(math.max(position.x), math.min(position.x))
    xExtent = math.abs(math.max(position.x) - math.min(position.x))
    yExtent = math.abs(math.max(position.y) - math.min(position.y))
    xMinimum = math.min(position.x)
    yMinimum = math.min(position.y)
    boxXs = []
    boxYs = []
    boxWidths = []
    boxHeights = []
    for box in boxes: 
        boxXs.append(box.x)
        boxYs.append(box.y)
        boxWidths.append(box.width)
        boxHeights.append(box.height)
    boxXs = math.array(boxXs)
    boxYs = math.array(boxYs)
    boxHeights = math.array(boxHeights)
    boxWidths = math.array(boxWidths)
    potentialRatios = math.array(potentialRatios)
    decomposed = zip(boxXs, boxYs, boxWidths, boxHeights, potentialRatios)
    for boxX, boxY, boxWidth, boxHeight, potentialRatio in decomposed: 
        xUpperBound : float = xMinimum + (xExtent * (boxX + boxWidth))
        yUpperBound : float = yMinimum + (yExtent * (boxY + boxHeight))
        xLowerBound : float = xMinimum + (xExtent * boxX)
        yLowerBound : float = yMinimum + (yExtent * boxY)
        xCondition = (position.x >= xLowerBound) & (position.x <= xUpperBound)
        yCondition = (position.y >= yLowerBound) & (position.y <= yUpperBound)
        #print(xLowerBound, xUpperBound, yUpperBound, yLowerBound)
        potential = math.where(xCondition & yCondition, potentialRatio * potentialHeight, potential)
    return potential 

