#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:31:36 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
import copy
import math
import numpy
import pandas
import pathlib
import time
import os
import sys
import scipy
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
        
        self.dataOutputRootFolder = pathlib.Path(dataOutputRootFolder)
        
        self.dataOutputFolder = self.dataOutputRootFolder / self.name
        
        self.responseVariable = responseVariable
        
        self.phase01CSVFile = self.dataOutputFolder / (self.name + "_phase01.csv")
        self.responseFromTrainingSimulationsCSVFile = self.dataOutputFolder / ( self.name + "_responseFromTrainingSimulations.csv")
        
        self.completeFile = self.dataOutputFolder / (self.name + "_complete.csv")
        self.statsFile = self.dataOutputFolder / (self.name + "_stats.csv")
        
        self.dataFile = self.dataOutputFolder / (self.name + "_DATA")
        
        self.filteredFile = self.dataOutputFolder / (self.name + "_filtered.csv")
        
        self.filterValue = filterValue
        
        self.responseIndicatorVariable = "responseIndicator"
        self.responseIndicator = responseIndicator
        
        self.fortranDataFolder = self.dataOutputFolder / ( "DATA" + "_" + self.responseVariable)
        
        self.numCores = multiprocessing.cpu_count()
        
        self.anomalyLimitLow = {}
        self.anomalyLimitHigh = {}
        self.anomalyQuantile = 0.02

        
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
        
        self.__init__saveFilteredDataRMode()
        
        # self.__init__saveOmittedData()
        
        # self.__init__saveNamelists()
        
        # self.__init__linkExecutables()
        
        # self.runTrain()
        
        # self.runPredictionSerial()
        
        self.collectSimulatedVSPredictedData()
        
        self._getSimulationCollection()
        
        self._filterGetOutlierDataFromLESoutput()
        
        self._fillUpDrflxValues()
        
        self._insertWPOSasInObservations()
        
        self._getAnomalyLimitsQuantile()
        self._getAnomalyLimitsConstant()
        
        self._getAnomalies()
        
        self._getLeaveOneOut()
        self._getLinearFit()
        
        self._finaliseStats()
        self._finaliseAnomalies()
        
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
        
        self.simulationFilteredCSV = self.simulationFilteredData[self.designVariableNames + [ self.responseVariable ]]
        
    def __init__saveFilteredDataFortranMode(self):
        
        self.simulationFilteredData.to_csv( self.dataFile, float_format="%017.11e", sep = " ", header = False, index = False, index_label = False )
        
    def __init__saveFilteredDataRMode(self):
        # self.simulationFilteredData["ind"] = range(self.simulationFilteredData.shape[0])
        # self.simulationFilteredData.set_index("ind", drop = True, inplace = True)
        self.simulationFilteredData.to_csv(self.filteredFile)
    
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
                
    def _getSimulationCollection(self):
        self.simulationCollection = InputSimulation.getSimulationCollection( self.simulationCompleteData )
        
    def _filterGetOutlierDataFromLESoutput(self):
    
        dataframe = self.simulationCompleteData
        collection = self.simulationCollection
        
        dataframe["lwpEndValue"] = dataframe.apply( lambda row: \
                                                   collection[ row["LABEL"] ].getTSDataset()["lwp_bar"].values[-1]*1000.,
                                                   axis = 1)
        
        dataframe["lwpRelativeChange"] = dataframe.apply( lambda row:\
                                                         row["lwpEndValue"] / row["lwp"],
                                                         axis = 1 )
          
        dataframe["cloudTopRelativeChange"] = dataframe.apply( lambda row: \
                                                              collection[ row["LABEL"] ].getTSDataset()["zc"].values[-1] / row["pblh_m"],
                                                              axis = 1)

        dataframe["cfracEndValue"] = dataframe.apply( lambda row: \
                                                     collection[ row["LABEL"] ].getTSDataset()["cfrac"].values[-1],
                                                     axis = 1)
            
    def _getAnomalyLimitsConstant(self):
        self.anomalyLimitLow["tpot_inv"] = 2.5
        self.anomalyLimitHigh["cloudTopRelativeChange"] = 1.1
        self.anomalyLimitHigh["lwpRelativeChange"] = 1.4
        
        self.anomalyLimitLow["tpot_pbl"] = 273
        self.anomalyLimitHigh["tpot_pbl"] =  300    
        
        
        
        if self.name[4:] == "Day":
            self.anomalyLimitLow["cos_mu"] =  math.cos(5*math.pi/12)
            self.anomalyLimitHigh["cos_mu"] =  math.cos(math.pi/12)
        else:
            self.anomalyLimitLow["cos_mu"] = 0
            self.anomalyLimitHigh["cos_mu"] = 0
    
    def _getAnomalyLimitsQuantile(self):
        dataframe = self.simulationCompleteData
        
        self.anomalyLimitLow["tpot_inv"] = dataframe["tpot_inv"].quantile(self.anomalyQuantile)
        self.anomalyLimitHigh["cloudTopRelativeChange"] = dataframe["cloudTopRelativeChange"].quantile(1-self.anomalyQuantile)
        self.anomalyLimitHigh["lwpRelativeChange"] = dataframe["lwpRelativeChange"].quantile(1-self.anomalyQuantile)
        self.anomalyLimitLow["q_inv"] = dataframe["q_inv"].quantile(self.anomalyQuantile)
        
        self.anomalyLimitLow["pblh"] =  dataframe["pblh"].quantile(0.05)
        self.anomalyLimitHigh["pblh"] = dataframe["pblh"].quantile(0.95)
        
        self.anomalyLimitLow["lwp"] =  dataframe["lwp"].quantile(0.05)
        self.anomalyLimitHigh["lwp"] =  dataframe["lwp"].quantile(0.95)
        
        if self.name[3] == "3":
            self.anomalyLimitLow["cdnc"] = dataframe["cdnc"].quantile(0.05)
            self.anomalyLimitHigh["cdnc"] = dataframe["cdnc"].quantile(0.95)
            
        
    def _getAnomalies(self):
        dataframe = self.simulationCompleteData
        
        for key in self.anomalyLimitLow:
            try:
                dataframe[key + "_low_tail"] = dataframe[key] < self.anomalyLimitLow[key]
            except KeyError:
                pass
        for key in self.anomalyLimitHigh:
            try:
                dataframe[key + "_high_tail"] = dataframe[key] > self.anomalyLimitHigh[key]
            except KeyError:
                pass
    
    def _finaliseAnomalies(self):
        low = pandas.DataFrame(self.anomalyLimitLow, index = ["low"]).transpose()
        high = pandas.DataFrame(self.anomalyLimitHigh, index = ["high"]).transpose()
        
        self.anomalyLimitsCombined = pandas.concat([low,high], axis = 1, verify_integrity=True)
        
        self.anomalyLimitsCombined.to_csv(self.dataOutputRootFolder / "anomalyLimits.csv")
        
    
    def _fillUpDrflxValues(self):
        
        if self.ID_prefix == "3":
            return
        
        dataframe = self.simulationCompleteData
        
        if numpy.abs(dataframe["drflx"].max() - dataframe["drflx"].min() ) > 10*Data.getEpsilon(): 
            return
        
        newCloudRadiativeValues = numpy.zeros(numpy.shape(dataframe["drflx"]))
            
        for emulInd, emul in enumerate(list(self.simulationCollection)):
            self.simulationCollection[emul].getPSDataset()
            self.simulationCollection[emul].setTimeCoordToHours()
            
            tstart = 2.5
            tend = 3.5
            tol_clw = 1e-5
            
            psDataTimeSliced = self.simulationCollection[emul].sliceByTimePSDataset(tstart, tend)
            
            numberOfCloudyColumns = 0
            
            cloudRadiativeWarmingAllColumnValues = 0.
            
            for timeInd, timeValue in enumerate(psDataTimeSliced["time"]):
                rflxTimeSlice = psDataTimeSliced["rflx"].isel( time = timeInd )
                
                liquidWaterTimeSlice = psDataTimeSliced["l"].isel( time = timeInd )
                
                psDataCloudyPointIndexes, = numpy.where( liquidWaterTimeSlice > tol_clw )
                if len(psDataCloudyPointIndexes > 0):
                    numberOfCloudyColumns += 1
                    
                    firstCloudyGridCell = psDataCloudyPointIndexes[0]
                    lastCloudyGridCell = psDataCloudyPointIndexes[-1]   
                    
                
                    cloudRadiativeWarmingAllColumnValues += rflxTimeSlice[firstCloudyGridCell] - rflxTimeSlice[lastCloudyGridCell]
                    
            ## end time for loop
            if numberOfCloudyColumns > 0:
                drflx = cloudRadiativeWarmingAllColumnValues.values / numberOfCloudyColumns
            else:
                drflx = 0.
            
            newCloudRadiativeValues[emulInd] = drflx
            
        ### end emul for loop
        dataframe["drflx"] = newCloudRadiativeValues
        
    def _insertWPOSasInObservations(self):
        
        
        
        dataframe = self.simulationCompleteData
        
        
        newWPOSValues = numpy.zeros(numpy.shape(dataframe["wpos"]))
        wposCheckingValues = numpy.zeros(numpy.shape(dataframe["wpos"]))
        
        wposAllColumnMean = 0.
        
        
        # indexGroup = self.simulationFilteredData.index.values
        # try:
        #     pool = multiprocessing.Pool(processes = self.numCores)
        #     for ind in pool.imap_unordered( self._runPrediction, indexGroup):
        #          pass
        # except Exception as e:
        #     print(e)
        #     pool.close()
        # pool.close()
        
        
            
        for emulInd, emul in enumerate(list(self.simulationCollection)):
            ncData = self.simulationCollection[emul].getNCDataset()
            self.simulationCollection[emul].setTimeCoordToHours()
            print(emul, "wpos", dataframe.loc[emul]["wpos"], "wpos2", dataframe.loc[emul]["w2pos"])
            
            tstart = 2.5
            tend = 3.5
            tol_clw = 1e-5
            
            timesList = []
            k = 0
            timeStartInd = Data.getClosestIndex( ncData.time.values, tstart )
            timeEndInd   = Data.getClosestIndex( ncData.time.values, tend ) + 1
            
            numberOfCloudyColumns = 0
            
            wposAllColumnValues = 0.
            
            w2posAllColumnValues = 0.
            
            for timeInd in range(timeStartInd, timeEndInd):
                
                liquidWaterTimeSlice = ncData["l"].isel( time = timeInd )
                for yCoordInd, yCoordValue in enumerate(liquidWaterTimeSlice["yt"]):
                    for xCoordInd, xCoordValue in enumerate(liquidWaterTimeSlice["xt"]):
                        print("time",k,"xCoordBegin");timesList.append( time.time() ); k+=1        
                        ncDataCloudyPointIndexes, = \
                            numpy.where( liquidWaterTimeSlice.isel(xt = xCoordInd, yt = yCoordInd) > tol_clw )
                        print("time",k,"ncDataCloudyPointIndexes");timesList.append( time.time() ); k+=1        
                        numberOfCloudyPoints = len(ncDataCloudyPointIndexes)
                        
                        if numberOfCloudyPoints > 0:
                            firstCloudyGridCell = ncDataCloudyPointIndexes[0]
                            middleCloydyGridCell = ncDataCloudyPointIndexes[numberOfCloudyPoints//2]
                        else:
                            continue
                        print("time",k,"gridCell");timesList.append( time.time() ); k+=1
                        wposAtFirstCloudyGridCell = ncData["w"].isel(time = timeInd, xt = xCoordInd, ym =yCoordInd, zt = firstCloudyGridCell)
                        print("time",k,"wpos gridCell");timesList.append( time.time() ); k+=1
                        # wposAtLowerHarfOfGridCell = wposTimeSlice.isel(xt = xCoordInd, ym = yCoordInd, zt = slice(firstCloudyGridCell, middleCloydyGridCell)).sum()
                        
                        if wposAtFirstCloudyGridCell > 0.:
                            
                            numberOfCloudyColumns += 1
                            
                            wposAllColumnValues +=  wposAtFirstCloudyGridCell
                            
                            w2posAllColumnValues += wposAtFirstCloudyGridCell**2
                            
                        print("time",k,"kalkyyl");timesList.append( time.time() ); k+=1
                        print("time",k,"xcoordEnd");timesList.append( time.time() ); k+=1
                        break
                    print("time",k,"ycoord");timesList.append( time.time() ); k+=1
                    break
                print("time",k,"timeIndEnd");timesList.append( time.time() ); k+=1
                break
                    
            ## end time for loop
            if numberOfCloudyColumns > 0:
                wposAllColumnMean = wposAllColumnValues.values / numberOfCloudyColumns
                newWPOSWeightedMean = w2posAllColumnValues / wposAllColumnValues
            
            print(emul, "wpos", newWPOSWeightedMean, "wpos2", newWPOSWeightedMean)
            newWPOSValues[ emulInd ] = newWPOSWeightedMean
            wposCheckingValues[ emulInd ] = wposAllColumnMean
            
        ### end emul for loop
        dataframe["wposWeighted"] = newWPOSValues
        dataframe["wposCheck"] = wposCheckingValues
    
        
    def _getLeaveOneOut(self):
        dataframe = self.simulationCompleteData
        
        dataframe["leaveOneOutIndex"]  = dataframe["wpos"] != -999
        
        dataframe = dataframe.loc[dataframe["leaveOneOutIndex"]]
        
        simulated = dataframe["wpos"].values
        emulated  = dataframe["wpos_Emulated"].values
        
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(simulated, emulated)

        rSquared = numpy.power(r_value, 2)
        
        self.leaveOneOutStats = [slope, intercept, r_value, p_value, std_err, rSquared]
        

    def _getLinearFit(self):
        
        dataframe = self.simulationCompleteData
        
        condition =  {}
        condition["wpos"] = dataframe["wpos"] != -999. 
        condition["tpot_inv"] = dataframe["tpot_inv"] > 5
        condition["lwpEndValue"] = dataframe["lwpEndValue"] > 10.
        condition["cfracEndValue"] = dataframe["cfracEndValue"] > 0.9
        condition["prcp"] =  dataframe["prcp"] < 1e-6
        
        
        dataframe["linearFitIndex"] = Data.getAllConditions(condition)
        
        dataframe = dataframe.loc[dataframe["linearFitIndex"]]        
            
        radiativeWarming  = dataframe["drflx"].values
        updraft =  dataframe["wpos"].values
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(radiativeWarming, updraft)
        
        rSquared = numpy.power(r_value, 2)
        
        self.linearFitStats = [slope, intercept, r_value, p_value, std_err, rSquared]
    

    
    def _finaliseStats(self):
        self.statsDataFrame = pandas.DataFrame(numpy.array([  self.leaveOneOutStats, self.linearFitStats ]),
                                               columns = ["slope", "intercept", "r_value", "p_value", "std_err", "rSquared"],
                                               index=["leaveOneOutStats", "linearFitStats"])
        
        self.statsDataFrame.to_csv( self.statsFile)        

    def finalise(self):
        self.simulationCompleteData.to_csv(self.completeFile)                
            
    
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
