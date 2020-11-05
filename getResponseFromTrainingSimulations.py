#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:31:36 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
import copy
import numpy
import pandas
import pathlib
import time
import os
import sys
import LES2emu
import f90nml
from subprocess import run
import multiprocessing
import functools
from shutil import copyfile

sys.path.append("../LES-03plotting")
from Data import Data
from InputSimulation import InputSimulation

class EmulatorData:
    def __init__(self, name : str, trainingSimulationRootFolder : list, dataOutputRootFolder : str, responseVariable : str, filterValue = -999, responseIndicator = 0):
        self.name = name
        
        self.ID_prefix = name[3:5]
        
        self.trainingSimulationRootFolder = pathlib.Path( "/".join(trainingSimulationRootFolder) )
        
        self.dataOutputFolder = pathlib.Path(dataOutputRootFolder) / self.name
        
        self.responseVariable = responseVariable
        
        self.phase01CSVFile = self.dataOutputFolder / (self.name + "_phase01.csv")
        self.responseFromTrainingSimulationsCSVFile = self.dataOutputFolder / ( self.name + "_responseFromTrainingSimulations.csv")
        
        self.dataFile = self.dataOutputFolder / (self.name + "_DATA")
        
        self.filterValue = filterValue
        
        self.responseIndicatorVariable = "responseIndicator"
        self.responseIndicator = responseIndicator
        
        self.fortranDataFolder = self.dataOutputFolder / ( "DATA" + "_" + self.responseVariable)
        
        self.numCores = multiprocessing.cpu_count()
        
        self._makeFolder(self.dataOutputFolder)
        self._makeFolder(self.fortranDataFolder)
        self._main()
        
    def _main(self):
        self.__init__getPhase01()
        
        self.__init__getDesignVariableNames()
        
        self.__init__getResponseFromTrainingSimulations()
        
        self.__init__setSimulationCompleteDataFromDesignAndTraining()
        
        self.__init__setResponseIndicatorFilteredData()
        
        self.__init__filterNan()
        
        self.__init__saveFilteredDataFortranMode()
        
        self.__init__saveOmittedData()
        
        self.__init__saveNamelists()
        
        self.__init__linkExecutables()
        
        self.runTrain()
        
        self.runPredictionSerial()
        
        self.collectSimulatedVSPredictedData()
        
        self.getSimulationCollection()
        
        self.finalise()
        
    def _makeFolder(self, folder):
        folder.mkdir( parents=True, exist_ok = True )
        
    def __init__getPhase01(self):
        self.phase01 = pandas.read_csv(self.phase01CSVFile, index_col=0)
    
    def __init__getDesignVariableNames(self):
        self.meteorologicalVariables = ["q_inv", "tpot_inv", "lwp", "tpot_pbl", "pblh"]
        
        self.microphysics = self.ID_prefix[0]
        
        self.timeOfDay = self.ID_prefix[1]
        
        if self.microphysics == "3":
            self.microphysicsVariables = ["cdnc"]
        elif self.microphysics == "4":
            self.microphysicsVariables = ["ks", "as", "cs", "rdry_AS_eff"]
        
        if self.timeOfDay == "D":
            self.timeOfDayVariable = ["cos_mu"]
        else:
            self.timeOfDayVariable = []
            
        self.designVariableNames = self.meteorologicalVariables + self.microphysicsVariables + self.timeOfDayVariable
    
    def __init__getResponseFromTrainingSimulations(self):
        
        if pathlib.Path( self.responseFromTrainingSimulationsCSVFile).is_file():
             self.responseFromTrainingSimulations = pandas.read_csv(self.responseFromTrainingSimulationsCSVFile, index_col=0)
        else:
             self.responseFromTrainingSimulations = LES2emu.GetEmu2Vars( str(self.trainingSimulationRootFolder), self.ID_prefix )
        
        self.responseFromTrainingSimulations.to_csv(self.responseFromTrainingSimulationsCSVFile)
        
        return self.responseFromTrainingSimulations
    
    def __init__setSimulationCompleteDataFromDesignAndTraining(self):
        
        self.simulationCompleteData = pandas.merge(self.phase01, self.responseFromTrainingSimulations,on = "ID", how="left")
        
    def __init__setResponseIndicatorFilteredData(self):
        self.simulationCompleteData[self.responseIndicatorVariable] = float(self.responseIndicator)        

    def __init__filterNan(self):
        
        self.simulationFilteredData = self.simulationCompleteData[ self.simulationCompleteData[ self.responseVariable ] != self.filterValue ]
        
        self.simulationFilteredData = self.simulationFilteredData[self.designVariableNames + [self.responseIndicatorVariable , self.responseVariable]]
        
    def __init__saveFilteredDataFortranMode(self):
        
        self.simulationFilteredData.to_csv( self.dataFile, float_format="%017.11e", sep = " ", header = False, index = False, index_label = False )
    
    def __init__saveOmittedData(self):
        
        #self.simulationFilteredData = self.simulationFilteredData.set_index( numpy.arange( self.simulationFilteredData.shape[0] ) )
        
        for ind in self.simulationFilteredData.index:
            folder = (self.fortranDataFolder / str(ind) )
            folder.mkdir( parents=True, exist_ok = True )
            
            train = self.simulationFilteredData.drop(ind)
            train.to_csv( folder / ("DATA_train_" + ind ), float_format="%017.11e", sep = " ", header = False, index = False )
            
            predict = self.simulationFilteredData.loc[ind].values
            numpy.savetxt( folder / ("DATA_predict_" + ind), predict, fmt="%017.11e", newline = " ")
            
    def __init__saveNamelists(self):
        
        
        
        for ind in self.simulationFilteredData.index:
            folder = (self.fortranDataFolder / str(ind) )
            folder.mkdir( parents=True, exist_ok = True )
            
            trainNML = {"inputoutput":
                    {"trainingDataInputFile":"DATA_train_" + ind,
                     "trainedEmulatorGPFile" :"out_" + ind + ".gp",
                     "separator": " ",
                     "debugFlag": False}}
            
            predictNML = {"inputoutput":
                      {"trainingDataInputFile":"DATA_train_" + ind,
	                   "trainedEmulatorGPFile":"out_" + ind + ".gp",
	                   "predictionDataInputFile":"DATA_predict_" + ind,
	                   "predictionOutputFile":"DATA_predict_output_" + ind,
	                   "separator":" ",
	                   "debugFlag":False}}
            
            with open(folder / "train.nml", mode="w") as trainNMLFile:
                f90nml.write(trainNML, trainNMLFile)
            with open(folder / "predict.nml", mode="w") as predictNMLFile:
                f90nml.write(predictNML, predictNMLFile)
                
                
    def __init__linkExecutables(self):
        for ind in self.simulationFilteredData.index:
            folder = (self.fortranDataFolder / str(ind) )
            folder.mkdir( parents=True, exist_ok = True )
            run(["ln","-sf", os.environ["GPTRAINEMULATOR"], folder / "gp_train"])
            run(["ln","-sf", os.environ["GPPREDICTEMULATOR"], folder / "gp_predict"])
    
        
        
    def _runTrain(self, ind):
        
        folder = (self.fortranDataFolder / str(ind) )
        print(" ")
        print(self.name,"checking training output, case", ind)
        os.chdir(folder)
        
        if not pathlib.Path("out_" + ind + ".gp").is_file():
            print("runTrain", folder)
            run([ "./gp_train"])
    
    def runTrain(self):
        
         
        indexGroup = self.simulationFilteredData.index.values
        try:
            pool = multiprocessing.Pool(processes = self.numCores)
            for ind in pool.imap_unordered( self._runTrain, indexGroup):
                 pass     
        except Exception as e:
            print(e)
            pool.close()
        pool.close()
    
    def _runPrediction(self,ind):
        folder = (self.fortranDataFolder / str(ind) )
        print(" ")
        print(self.name, "checking prediction output, case", ind)
        os.chdir(folder)
        outputfile = pathlib.Path("DATA_predict_output_" + ind)
        
        fileExists = outputfile.is_file()
        
        if fileExists:
            fileIsEmpty = numpy.isnan(self._readPredictedData(".", ind))
            if fileIsEmpty:
                outputfile.unlink() #remove file
            
        if not fileExists:
            print("runPrediction", folder)
            run([ "./gp_predict"])
    
    def runPredictionParallel(self):
        
        indexGroup = self.simulationFilteredData.index.values
        
        try:
            pool = multiprocessing.Pool(processes = self.numCores)
            for ind in pool.imap_unordered( self._runPrediction, indexGroup):
                 pass
        except Exception as e:
            print(e)
            pool.close()
        pool.close()
        
    def runPredictionSerial(self):
        for ind in self.simulationFilteredData.index.values:
            self._runPrediction(ind)
        
            
    def collectSimulatedVSPredictedData(self):
        datadict = {self.responseVariable + "_Simulated": [],
                    self.responseVariable + "_Emulated": []}
        
        for ind in self.simulationFilteredData.index:
            folder = (self.fortranDataFolder / str(ind) )
            
            simulated = pandas.read_csv( folder / ("DATA_predict_" + ind), delim_whitespace = True, header = None).iloc[0,-1]
            emulated  = self._readPredictedData(folder, ind)
            

            datadict[self.responseVariable + "_Simulated"].append(simulated)
            datadict[self.responseVariable + "_Emulated"].append(emulated)
        
        
        
        for key in datadict:
            
            self.simulationFilteredData[key] = datadict[key]
        
        self._checkIntegrety()
        
        self.simulationFilteredData.drop(columns = self.responseIndicatorVariable, inplace = True)
        self.simulationCompleteData.drop(columns = self.responseIndicatorVariable, inplace = True)
        
        if self.allIsWell:
            self.simulationFilteredData.drop(columns = self.responseVariable + "_Simulated", inplace = True )
        
        self.simulationCompleteData = self.simulationCompleteData.merge( self.simulationFilteredData, on = ["ID", "wpos"] + self.designVariableNames, how = "left")
        
        

    
    def _readPredictedData(self, folder, ind):
        try:
            data = pandas.read_csv( pathlib.Path(folder) / ("DATA_predict_output_" + ind), delim_whitespace = True, header = None).iloc[0,0]
        except pandas.errors.EmptyDataError:
            data = numpy.nan
        return data
    
    def _checkIntegrety(self):
        self.allIsWell = True
        for ind in self.simulationFilteredData.index:
            relativeError = numpy.abs( (self.simulationFilteredData.loc[ind][self.responseVariable] - self.simulationFilteredData.loc[ind][self.responseVariable + "_Simulated"]) 
                      / self.simulationFilteredData.loc[ind][self.responseVariable]) * 100.
            if relativeError > 0.1:
                print("case", ind, "relative error", relativeError)
                self.allIsWell = False
                
    def getSimulationCollection(self):
        self.simulationCollection = InputSimulation.getSimulationCollection( self.simulationCompleteData )

    def finalise(self):
        self.simulationCompleteData.to_csv(self.dataOutputFolder / (self.name + "_complete.csv"))                
            
    
def main(responseVariable):
    
    rootFolderOfEmulatorSets = os.environ["EMULATORDATAROOTFOLDER"]
    rootFolderOfDataOutputs = "/home/aholaj/Data/EmulatorManuscriptData2"# os.environ["EMULATORPOSTPROSDATAROOTFOLDER"]
   
    ###########
    emulatorSets = {"LVL3Night" : EmulatorData("LVL3Night",
                                             [rootFolderOfEmulatorSets,  "case_emulator_DESIGN_v3.0.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_night"],
                                             rootFolderOfDataOutputs,
                                             responseVariable
                                             ),
                "LVL3Day"   :  EmulatorData( "LVL3Day",
                                            [rootFolderOfEmulatorSets, "case_emulator_DESIGN_v3.1.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_day"],
                                            rootFolderOfDataOutputs,
                                            responseVariable
                                            ),
                "LVL4Night" :  EmulatorData("LVL4Night",
                                            [rootFolderOfEmulatorSets, "case_emulator_DESIGN_v3.2_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_night"],
                                            rootFolderOfDataOutputs,
                                            responseVariable
                                            ),
                "LVL4Day"   : EmulatorData("LVL4Day",
                                            [rootFolderOfEmulatorSets, "case_emulator_DESIGN_v3.3_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_day"],
                                            rootFolderOfDataOutputs,
                                            responseVariable
                                            )
                }
        # emulatorSets[key].runTrain()
        
        # emulatorSets[key].runPredictionSerial()
        
        emulatorSets[key].collectSimulatedVSPredictedData()
    
    
if __name__ == "__main__":
    start = time.time()
    try:
        responseVariable = sys.argv[1]
    except IndexError:
        responseVariable = "wpos"
    
    print("responseVariable: ",responseVariable)
    main(responseVariable)
    end = time.time()
    print("Script completed in " + str(round((end - start),0)) + " seconds")
