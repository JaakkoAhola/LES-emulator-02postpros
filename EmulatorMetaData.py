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


class EmulatorMetaData:
    def __init__(self, configFile):
        
        self.configFile = configFile
        
        self.initConfigFile()
        
        self.nameFilterIndex()
        
        self.initResponseVariableDerivatives()
        
    def initConfigFile(self):
        self._readConfigFile()
        
        self._setUpParametersFromConfigFile()
        
        self._testConfigFile()
        
        self._testConfigFileResponseVariableIsInFilter()
        

    def _setUpParametersFromConfigFile(self):
        self.responseVariable = self.configFile["responseVariable"]

        self.useFortran = self.configFile["useFortran"]
            
        self.override = self.configFile["override"]
        
        self.filteringVariablesWithConditions = self.configFile["filteringVariablesWithConditions"]
        
        
        self.boundOrdo = list(map(float, self.configFile["boundOrdo"]))
        
        self.responseIndicator = self.configFile[self.responseIndicatorVariable]
        
        self.bootstrappingParameters = self.configFile[ "bootStrap" ]
        
        self.runLeaveOneOut = self.configFile["runLeaveOneOut"]
        self.runBootStrap = self.configFile["runBootStrap"]
        self.runPostProcessing = self.configFile["runPostProcessing"]    
        
        self.kFlodSplits = self.configFile["kFlodSplits"]
        

    def _readConfigFile(self):
        self.configFile = FileSystem.readYAML(self.configFile)
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
        
        self.predictionVariableList = [self.emulatedVariable, self.linearFitVariable, self.correctedLinearFitVariable]
        
    def joinList(self, lista):
        return "_".join(lista)
