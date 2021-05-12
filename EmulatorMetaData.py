#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 12:06:18 2021

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
import sys
import os
sys.path.append(os.environ["LESMAINSCRIPTS"])
from FileSystem import FileSystem
import pathlib

class EmulatorMetaData:
    def __init__(self, locationsFile):
        
        self.initLocationsFile(locationsFile)
        
        self.initConfigFile()
        
        self.nameFilterIndex()
        
        self.initResponseVariableDerivatives()
        
        self.initDesignVariables()
    
    def initLocationsFile(self, locationsFile):
        
        assert( pathlib.Path(locationsFile).is_file() )
        
        self.locationsFile = FileSystem.readYAML(locationsFile)
        
        try:
            self.configFile = FileSystem.readYAML(self.locationsFile["configFile"])
        except KeyError:
            sys.exit("configFile not given. Exiting.")
        
        try:
            self.trainingSimulationRootFolder = pathlib.Path(self.locationsFile["trainingSimulationRootFolder"])
        except KeyError:
            pass
        
        try:
            self.postProsDataRootFolder = pathlib.Path(self.locationsFile["postProsDataRootFolder"])
        except KeyError:
            sys.exit("dataOutputRootFolder not given. Exiting.")
        
        try:
            self.figureFolder = pathlib.Path(self.locationsFile["figureFolder"])
        except KeyError:
            pass
        
        try:
            self.tableFolder = pathlib.Path(self.locationsFile["tableFolder"])
        except KeyError:
            pass
        
    
    def initConfigFile(self):
        
        self._setUpParametersFromConfigFile()
        
        self._testConfigFile()
        
        self._testConfigFileResponseVariableIsInFilter()
        

    def _setUpParametersFromConfigFile(self):
        self.responseVariable = self.configFile["responseVariable"]

        self.useFortran = self.configFile["useFortran"]
            
        self.override = self.configFile["override"]
        
        self.filteringVariablesWithConditions = self.configFile["filteringVariablesWithConditions"]
        
        
        self.boundOrdo = list(map(float, self.configFile["boundOrdo"]))
        
        self.responseIndicatorVariable = "responseIndicator"
        
        self.responseIndicator = self.configFile[self.responseIndicatorVariable]
        
        self.bootstrappingParameters = self.configFile[ "bootStrap" ]
        
        self.runLeaveOneOut = self.configFile["runLeaveOneOut"]
        self.runBootStrap = self.configFile["runBootStrap"]
        self.runPostProcessing = self.configFile["runPostProcessing"]    
        
        self.kFlodSplits = self.configFile["kFlodSplits"]
        

    def _testConfigFile(self):
        for key in ["responseVariable", "filteringVariablesWithConditions", self.responseIndicatorVariable]:
            assert(key in self.configFile)        
            
    def _testConfigFileResponseVariableIsInFilter(self):
        assert(self.responseVariable in self.filteringVariablesWithConditions)
        
    def nameFilterIndex(self):
        st="responseVariable" + self.configFile["responseVariable"] + ":"
        for key in sorted(self.configFile["filteringVariablesWithConditions"]):
            st = st + key + self.configFile["filteringVariablesWithConditions"][key] + ";"
        
        self.filterIndex = st + "."
    
    def initResponseVariableDerivatives(self):
        
        self.simulatedVariable = self.joinList([self.filterIndex, "Simulated"])

        self.emulatedVariable = self.joinList([self.filterIndex, "Emulated"])

        self.linearFitVariable = self.joinList([self.filterIndex, "LinearFit"])
        
        self.correctedLinearFitVariable = self.joinList([self.filterIndex, "CorrectedLinearFit"])
        
        self.predictionVariableList = [ self.linearFitVariable, self.correctedLinearFitVariable, self.emulatedVariable]
        
    def initDesignVariables(self):
        self.meteorologicalVariables = ["tpot_pbl", "tpot_inv", "q_inv", "lwp", "pblh",  ]
        
        self.microphysicsVariablesPool = {"SB" : ["cdnc"],
                                          "SALSA": ["ks", "as", "cs", "rdry_AS_eff"]}
        
        self.solarZenithAngle = ["cos_mu"]
        
        self.designVariablePool = self.meteorologicalVariables + [item for sublist in  list(self.microphysicsVariablesPool.values()) for item in sublist] + self.solarZenithAngle
        
    def joinList(self, lista):
        return "_".join(lista)
