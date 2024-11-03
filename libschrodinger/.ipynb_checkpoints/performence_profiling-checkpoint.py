from typing import Dict, Callable, List
from time import monotonic

PerformenceLogType = Dict[str, float]
LogFunctionType = Callable[[PerformenceLogType, str], PerformenceLogType] 

def performenceLog(log : Dict[str, List[float]], stepLabel : str): 
    if stepLabel not in log: 
        log[stepLabel] = []
    log[stepLabel].append(monotonic())
