from libschrodinger.crank_nicolson_2d import Simulator, SimulationProfile, makeLinspaceGrid, totalProbabilityInRegion, animateImages
from libschrodinger.potentials import constantPotentials
import numpy as np
from typing import Tuple
from typing import List
from typing import Dict
from functools import partial
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd

def recordTotalLengthWiseProbabilities(
            simulator : Simulator, 
            regionLengths : List[float], 
            regionLabels : Tuple[str], 
            math = np, 
            regionLabelPrepend = "TotalProbability::"
        ) -> pd.DataFrame:
    assert len(regionLabels) == len(regionLengths)
    currentPosition : float = 0
    regionalProbabilities : List[np.array] = []
    regionalCutFrames : List[np.array] = []
    for ii in range(len(regionLengths)): 
        regionLength = regionLengths[ii]
        probabilities, cutFrames = totalProbabilityInRegion(
                simulator.probabilities, 
                simulator.grid.pointCount, 
                simulator.spaceStep, 
                currentPosition, 
                0, 
                regionLength, 
                1, 
                math
            )
        regionalProbabilities.append(probabilities)
        regionalCutFrames.append(cutFrames)
        currentPosition += regionLength
    totalProbabilities = {
            (regionLabelPrepend + "Probabilities::" + regionLabels[ii]) : regionalProbabilities[ii] \
            for ii in range(len(regionLengths))
        }
    return totalProbabilities, regionalCutFrames


def logConstantMeasurementRegionSimulation(
            frameCount : int, 
            baseName : str, 
            length : float, 
            simulator : Simulator, 
            simulationCount : int, 
            allData : Dict, 
            constantRegionLengths : List[float], 
            constantRegionLabels : List[str], 
            showWhenSimulationDone : bool = False, 
            math = np, 
            basePath = None, 
            animationInterval : int = 30, 
            colorMap : str = "hot"
        ):
    basePath = basePath if basePath else Path.cwd() / baseName
    if showWhenSimulationDone == True: 
        print("Simulation " + str(simulationCount) + ": done processing probabilities.")
    if showWhenSimulationDone == True: 
        print("Simulation " + str(simulationCount) + ": logging.")
    name = baseName + str(simulationCount) + "::"
    if showWhenSimulationDone == True: 
        print("Saving Video of " + name[:-2])
    videoPath = basePath / str(simulationCount)
    videoPath.mkdir(parents = True, exist_ok = True)
    totalProbabilities, cutFrames = recordTotalLengthWiseProbabilities(
            simulator, 
            constantRegionLengths, 
            constantRegionLabels, 
            math, 
            name
        )
    waveAnimation = animateImages(
            length, 
            simulator.probabilities, 
            animationInterval, 
            0, 
            math.max(simulator.probabilities), 
            constantRegionLengths, 
            [1] * len(constantRegionLengths), 
            colorMap = colorMap
        )
    waveAnimation.save(str(videoPath / (str(simulationCount) + ".gif")))
    plt.close()
    plt.figure()
    plt.imshow(simulator.potentials[0])
    plt.savefig(str(videoPath / (str(simulationCount) + "Potential.png")))
    plt.close()
    totalProbabilities[name + "FrameCount"] = frameCount
    totalProbabilities[name + "SpaceStep"] = simulator.profile.spaceStep
    totalProbabilities[name + "TimeStep"] = simulator.profile.timeStep
    totalProbabilities[name + "PointCount"] = simulator.profile.grid.pointCount
    for ii in range(len(constantRegionLabels)): 
        print("Saving Video of " + constantRegionLabels[ii])
        totalProbabilities[name + "RegionLength::" + constantRegionLabels[ii]] = constantRegionLengths[ii]
        cutAnimation = animateImages(
                length * constantRegionLengths[ii], 
                cutFrames[ii], 
                animationInterval, 
                0, 
                math.max(cutFrames[ii]), 
                colorMap = "hot"
            )
        cutAnimation.save(str(videoPath / (constantRegionLabels[ii] + ".gif")))
        plt.close()
    allData |= totalProbabilities
    if showWhenSimulationDone == True: 
        print("Done logging " + name[:-2])
    simulationCount += 1
    if showWhenSimulationDone == True: 
        print("Producing Simulation final output CSV")
    return allData

def constantSimulationProfiles( 
            initialWaveFunction, 
            spatialStep : float, 
            temporalStep : float,
            length : float, 
            regionLengthRatios : List[float], 
            regionPotentialRatios : List[List[float]], 
            potentialHeight : float, 
            pointCount : int, 
            simulateControl : bool, 
            math = np, 
            gpuAccelerated = False, 
            edgeBound = False, 
            useDense = False, 
            courantNumber = 1.0, 
        ) -> List[SimulationProfile]:
    if simulateControl == True: 
        regionPotentialRatios.append([0.0 for ii in range(len(regionPotentialRatios[0]))])
    profiles : List[SimulationProfile] = []
    potentialFunction = lambda potentialRatios, position, time : constantPotentials(
                position, 
                math.array(regionLengthRatios), 
                potentialRatios, 
                potentialHeight, 
                math
        )
    for potentialRatios in regionPotentialRatios: 
        profile = SimulationProfile(
            makeLinspaceGrid(pointCount, length, 2, False, float, math), 
            initialWaveFunction, 
            partial(potentialFunction, potentialRatios), 
            temporalStep, 
            spatialStep, 
            constantPotential = True, 
            gpuAccelerated = gpuAccelerated, 
            edgeBound = edgeBound, 
            useDense = useDense, 
            courantNumber = courantNumber, 
            length = length
        )
        profiles.append(profile)
    return profiles

def recordConstantRegionSimulations(
            profiles : List[SimulationProfile], 
            frames : int, 
            baseName : str, 
            measurmentRegions : List[float], 
            showWhenSimulationDone = False, 
            discardSimulations = True, 
            constantRegionLabels : List[str] = None, 
            basePath = None, 
            animationInterval = 30, 
            showBar : bool = False, 
            showFPS : bool = False, 
            showTotalTime : bool = False, 
            math = np
        ):
    simulations : List[Simulator] = []
    simulationCount : int = 0
    allData = {}
    constantRegionLabels \
            = ["Region" + str(ii) for ii in range(len(constantRegionLabels))] \
            if constantRegionLabels == None else constantRegionLabels 
    for profile in profiles: 
        simulator = Simulator(profile)
        simulator.simulate(frames, showBar, showFPS, showTotalTime)
        if showWhenSimulationDone == True: 
            print("Simulation " + str(simulationCount) + " is done, processing probabilities.")
        probabilities, probabilityDecibles = simulator.processProbabilities()
        if discardSimulations == False: 
            simulations.append(simulator)
        allData = logConstantMeasurementRegionSimulation(
                frames, 
                baseName, 
                simulator.profile.length, 
                simulator, 
                simulationCount, 
                allData, 
                measurmentRegions, 
                constantRegionLabels, 
                showWhenSimulationDone, 
                math, 
                basePath, 
                animationInterval
            )
        simulationCount += 1
    return allData, simulations

