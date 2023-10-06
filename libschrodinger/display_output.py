from typing import List, Dict, Tuple
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from time import monotonic
import sys
from libschrodinger.utility import *

def printWithProgressBar(
            step, 
            timePoints : int, 
            progressBarLength : int = 100, 
            printProgress : bool = False, 
            showTotalTime : bool = False, 
            showStepTime : bool = False, 
            detailedProgress : bool = False
        ): 
    performenceStartTime = monotonic()
    performenceAverageStepTime = 1
    deltaProgress = 1
    lastProgressLength = 0
    messageLength = 1
    progressOffset : int = timePoints % progressBarLength
    if printProgress == True: 
        if detailedProgress == False: 
            sys.stdout.write("[" + ("=" * progressBarLength) + "]\n")
        sys.stdout.write("[")
        if detailedProgress == True: 
            sys.stdout.write("]")
    for ii in range(1, timePoints): 
        previousPerformenceTime = monotonic()
        step(ii)
        progress = round((ii / timePoints) * progressBarLength)
        if printProgress == True: 
            update = ((progress - lastProgressLength) // deltaProgress)
            if detailedProgress == True: 
                percent = " " + "{0:.3g}".format((ii / timePoints) * 100) + "%"
                frames = " (" + str(ii) + "/" + str(timePoints) + ")"
                performence = " {0:.3g} fps".format(1 / performenceAverageStepTime)
                remainingTime = ", est. {0:.3g}s remain".format((timePoints - ii) / (1 / performenceAverageStepTime))
                message = frames + percent + performence + remainingTime + "]"
                for jj in range(messageLength): 
                    sys.stdout.write("\b")
                    sys.stdout.flush()
            #if (progress - lastProgressLength) > deltaProgress: 
            sys.stdout.write("-" * update)
            if detailedProgress == True: 
                sys.stdout.write(message)
                messageLength = len(message)
            sys.stdout.flush()
            lastProgressLength += update
        performenceAverageStepTime = (performenceAverageStepTime \
                + (monotonic() - previousPerformenceTime)) / 2.0
    if printProgress == True: 
        sys.stdout.write("]\n")
        sys.stdout.flush()
    if showTotalTime == True: 
        print("Total Time: ", monotonic() - performenceStartTime)
    if showStepTime == True: 
        print("Frames Per Second: ", 1 / performenceAverageStepTime)

animateImagesDefaultTitleFormatString = "Frame: {}"

def animateImages(
            length : float, 
            images : List[np.array], 
            interval = 1, 
            minimumValue = None, 
            maximumValue = None, 
            lengthRatios = None, 
            potentialRatios = None, 
            baseAlpha : float = .08, 
            colorMap : str = "viridis", 
            showFrame : bool = True, 
            titleFormatString : str = animateImagesDefaultTitleFormatString 
        ): 
    animationFigure = plt.figure()
    animationAxis = animationFigure.add_subplot(xlim=(0, length), ylim=(0, length))
    animationFrame = animationAxis.imshow(
            asNumPyArray(images[0]), 
            extent=[0, length, 0, length], 
            vmin = minimumValue, 
            vmax = maximumValue, 
            zorder = 1, 
            cmap = colorMap
        )
    if showFrame == True: 
        animationAxis.set_title(titleFormatString.format(str(0)))
    elif titleFormatString != None: 
        animationAxis.set_title(titleFormatString)
    if lengthRatios and potentialRatios: 
        constantPotentialRectangles(
                animationAxis, 
                length, 
                lengthRatios, 
                potentialRatios, 
                baseAlpha = baseAlpha
            )
    def animateFrame(frameIndex): 
        if showFrame == True: 
            animationAxis.set_title(titleFormatString.format(str(frameIndex)))
        elif titleFormatString != None: 
            animationAxis.set_title(titleFormatString)
        animationFrame.set_data(asNumPyArray(images[frameIndex]))
        animationFrame.set_zorder(1)
        return animationFrame,
    animation = FuncAnimation(
            animationFigure, 
            animateFrame, 
            interval = interval, 
            frames = np.arange(0, len(images), 2), 
            repeat = True, 
            blit = 0
        )
    return animation

def constantPotentialRectangles(
            axis, 
            pointCount : int, 
            lengthRatios : List[float], 
            potentialRatios : List[float], 
            baseAlpha : float = .08, 
            color : str = "w", 
            zorder : int = 50
        ): 
    displayRectangles = []
    currentPosition = 0
    for ii in range(len(lengthRatios)): 
        xExtent = pointCount * lengthRatios[ii]
        displayRectangles.append(Rectangle(
                (currentPosition, 0), 
                xExtent, 
                pointCount, 
                color = color, 
                zorder = zorder, 
                alpha = potentialRatios[ii] * baseAlpha
            ))
        axis.add_patch(displayRectangles[-1])
        currentPosition += xExtent
    return displayRectangles

