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
        self._handleConfigFile()
        
        self.responseVariable = self.configFile["responseVariable"]

        self.useFortran = self.configFile["useFortran"]
            
        self.override = self.configFile["override"]
        
        self.filteringVariablesWithConditions = self.configFile["filteringVariablesWithConditions"]
        
        self.testIfResponseVariableIsInFilter()
        
        self.boundOrdo = list(map(float, self.configFile["boundOrdo"]))
        
        self.responseIndicator = self.configFile[self.responseIndicatorVariable]
        
        self.bootstrappingParameters = self.configFile[ "bootStrap" ]
        
        self.runEmulator = self.configFile["runEmulator"]
        self.runBootStrap = self.configFile["runBootStrap"]
        self.runPostProcessing = self.configFile["runPosProcessing"]

    def initResponseVariableDerivatives(self):
        
        self.simulatedVariable = self.jointList([self.filterIndex, "Simulated"])

        self.emulatedVariable = self.jointList([self.filterIndex, "Emulated"])

        self.linearFitVariable = self.jointList([self.filterIndex, "LinearFit"])
        
    def joinList(self, lista):
        return "_".join(lista)

    def _handleConfigFile(self):
        self._readConfigFile()
        self._testConfigFile()

    def _readConfigFile(self):
        self.configFile = FileSystem.readYAML(self.configFile)
    def _testConfigFile(self):
        for key in ["responseVariable", "filteringVariablesWithConditions", self.responseIndicatorVariable]:
            assert(key in self.configFile)        
            
    def testIfResponseVariableIsInFilter(self):
        assert(self.responseVariable in self.filteringVariablesWithConditions)
        
    def nameFilterIndex(self):
        st="responseVariable" + self.configFile["responseVariable"] + ":"
        for key in sorted(self.configFile["filteringVariablesWithConditions"]):
            st = st + key + self.configFile["filteringVariablesWithConditions"][key] + ";"
        
        self.filterIndex = st + "."
    