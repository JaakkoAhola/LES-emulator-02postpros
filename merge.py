#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:58:24 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
import numpy
import time
import os
import pandas
import pathlib
import scipy
import sys

sys.path.append("../LES-03plotting")
from Data import Data
from InputSimulation import InputSimulation

class Merge:
    def __init__(self, name : str, dataOutputRootFolder : str):
        
        self.name = name
        
        self.ID_prefix = name[3:5]
        
        
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
            
        
            
        self.otherCommonVariables = ["ID", "wpos"]
        
        self.variables  = self.otherCommonVariables + self.meteorologicalVariables + self.microphysicsVariables + self.timeOfDayVariable
        
        self.rootFolder = pathlib.Path(dataOutputRootFolder)
        
        self.subFolder = self.rootFolder / name
        
        
        self.responseFile = self.subFolder / (name + "_responseFromTrainingSimulations.csv")
        
        self.simulatedVSPredictedFile = self.subFolder / (name + "_simulatedVSPredictedData.csv")
        
        self.bootStrapSampleSize = 100
        self.bootStrapNumberOfSamples = 10000
        
        self._initMetaData()
        
        self._initResponse()
        
        self._initSimulatedVSPredictedFile()
        
        self._initMerge()

        self._initGetSimulationCollection()
        
        self._getLeaveOneOut()
        
        self._getLinearFit()
        
        self._finaliseStats()
        
        self._finaliseSaveMerged()
        
    def _initMetaData(self):
        self.metaDataFrame = pandas.read_csv(self.rootFolder / (self.name + ".csv"))
        
        self.metaDataFrame = self.metaDataFrame.drop(columns= ["ID.1", "ID.1.1"])
        
    def _initResponse(self):
        self.responseDataFrame = pandas.read_csv(self.responseFile)
        
        self.responseDataFrame["ID"] = self.responseDataFrame.apply(lambda row: self.ID_prefix +"_" + str(int(row["i"])).zfill(3), axis = 1)
        
        self.responseDataFrame = self.responseDataFrame.drop(columns="i")
        
    def _initSimulatedVSPredictedFile(self):
        self.simulatedVSPredictedDataFrame = pandas.read_csv(self.simulatedVSPredictedFile)
        
        self.simulatedVSPredictedDataFrame["ID"] = self.simulatedVSPredictedDataFrame.apply( \
                                                            lambda row: self.ID_prefix +"_" + str(int(row["designCase"] + 1)).zfill(3), axis = 1)
        
        self.simulatedVSPredictedDataFrame = self.simulatedVSPredictedDataFrame.drop(columns="designCase")
        
    def _initMerge(self):
        
        self.mergedDataFrame = self.metaDataFrame.merge( self.responseDataFrame, on="ID", how = "left")
        
        
        
        self.mergedDataFrame = self.mergedDataFrame.merge(self.simulatedVSPredictedDataFrame,
                                                          on = self.variables,
                                                          how="left")

        self.mergedDataFrame.set_index("ID", drop=False, inplace = True )
        
        
    def _initGetSimulationCollection(self):
        self.simulationCollection = InputSimulation.getSimulationCollection( self.mergedDataFrame )
        
    def _initLoadTimeSeriesLESData(self):
        # load ts-datasets and change their time coordinates to hours
        keisseja = 10  # TODO REMOVE THIS
        for emul in list(self.simulationCollection)[:keisseja]:
                
            self.simulationCollection[emul].getTSDataset()
            self.simulationCollection[emul].setTimeCoordToHours()
                
    def _initLoadProfileLESData(self):
        # load ts-datasets and change their time coordinates to hours
        keisseja = 10  # TODO REMOVE THIS
        for emul in list(self.simulationCollection)[:keisseja]:
            self.simulationCollection[emul].getPSDataset()
            self.simulationCollection[emul].setTimeCoordToHours()
            
    def _filterGetOutlierDataFromLESoutput(self):
    
        dataframe = self.mergedDataFrame                
                    
        lwpRelativeChange = []
        cloudTopRelativeChange = []
        lwpEndValue = []
        
        for emulInd, emul in enumerate(self.simulationCollection):
            
            
            lwpStart = dataframe.loc[emul]["lwp"]
            print(lwpStart)
            lwpEnd = self.simulationCollection[emul].getTSDataset()["lwp_bar"].values[-1]*1000.
            print(lwpEnd)
            
            cloudTopStart = dataframe.loc[emul]["pblh_m"]
            cloudTopEnd = self.simulationCollection[emul].getTSDataset()["zc"].values[-1]
            
            
            lwpRelativeChange.append(lwpEnd / lwpStart)
            cloudTopRelativeChange.append(cloudTopEnd / cloudTopStart)
            lwpEndValue.append(lwpEnd)
            
        dataframe["lwpRelativeChange"] = lwpRelativeChange
        dataframe["cloudTopRelativeChange"] = cloudTopRelativeChange
        dataframe["lwpEndValue"] = lwpEndValue
            
    def _fillUpDrflxValues(self):
        
        if self.ID_prefix == "3":
            return
        
        responseData = self.mergedDataFrame
        
        if numpy.abs(responseData["drflx"].max() - responseData["drflx"].min() ) > 10*Data.getEpsilon(): 
            return
        else:
            print("let us refill missing drflx values", self.name)
        
        newCloudRadiativeValues = numpy.zeros(numpy.shape(responseData["drflx"]))
            
        for emulInd, emul in enumerate(list(self.simulationCollection)):
            self.simulationCollection[emul].getPSDataset()
            self.simulationCollection[emul].setTimeCoordToHours()
            
            tstart = 2.5
            tend = 3.5
            tol_clw=1e-5
            
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
        responseData["drflx"] = newCloudRadiativeValues
    
    def _getLeaveOneOut(self):
        dataframe = self.mergedDataFrame
        
        dataframe["leaveOneOutIndex"]  = dataframe["wpos"] != -999
        
        dataframe = dataframe.loc[dataframe["leaveOneOutIndex"]]
        
        simulated = dataframe["wpos_Simulated"].values
        emulated  = dataframe["wpos_Emulated"].values
        
        print(simulated)
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(simulated, emulated)
        print(slope)
        rSquared = numpy.power(r_value, 2)
        
        self.leaveOneOutStats = [slope, intercept, r_value, p_value, std_err, rSquared]
        
        print(self.leaveOneOutStats)

    def _getLinearFit(self):
        
        dataframe = self.mergedDataFrame
        
        dataframe = dataframe[ dataframe["wpos"] != -999. ]
        dataframe = dataframe[ dataframe["prcp"] < 1e-6 ] #NEW
        
            
        radiativeWarming  = dataframe["drflx"].values
        updraft =  dataframe["wpos"].values
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(radiativeWarming, updraft)
        
        rSquared = numpy.power(r_value, 2)
        
        self.linearFitStats = [slope, intercept, r_value, p_value, std_err, rSquared]
    
    def _finaliseStats(self):
        self.statsDataFrame = pandas.DataFrame(numpy.array([  self.leaveOneOutStats, self.linearFitStats ]),
                                               columns = ["slope", "intercept", "r_value", "p_value", "std_err", "rSquared"],
                                               index=["leaveOneOutStats", "linearFitStats"])
        
        self.statsDataFrame.to_csv( self.subFolder / ( self.name + "_stats.csv" ))
        
    def _finaliseSaveMerged(self):
        
        self.mergedDataFrame.to_csv( self.rootFolder /  (self.name +  "_merged.csv"), index_label = "ID")
        
    
    def getBootstrapLinRegress(self, dataframe, xName, yName):
        bootStrapRSquared = numpy.zeros(self.bootStrapNumberOfSamples)            
        for state in range(self.bootStrapNumberOfSamples):
            dataframeSample = dataframe.sample( n=self.bootStrapSampleSize, random_state=state )
            slopeSample, interceptSample, r_valueSample, p_valueSample, std_errSample = scipy.stats.linregress(dataframeSample[xName], dataframeSample[yName])            
            bootStrapRSquared[state] = numpy.power(r_valueSample,2)
        
        return numpy.mean(bootStrapRSquared), numpy.std(bootStrapRSquared)
    
    def getBootstrapMeanAverage(self, dataframe, xName, yName):
        meanAbsErrorList = numpy.zeros(self.bootStrapNumberOfSamples)            
        for state in range(self.bootStrapNumberOfSamples):
            dataframeSample = dataframe.sample( n=self.bootStrapSampleSize, random_state=state )
            meanAbsError = numpy.mean(numpy.abs( dataframeSample[xName] - dataframeSample[yName] ) )
            meanAbsErrorList[state] = meanAbsError
        
        return numpy.mean(meanAbsErrorList), numpy.std(meanAbsErrorList)
    
        
    
def main():
    
    rootFolderOfDataOutputs = os.environ["EMULATORPOSTPROSDATAROOTFOLDER"]
   

    allData = {"LVL3Night" : Merge("LVL3Night",rootFolderOfDataOutputs),
            "LVL3Day"   :  Merge( "LVL3Day",rootFolderOfDataOutputs),
            "LVL4Night" :  Merge("LVL4Night",rootFolderOfDataOutputs),
            "LVL4Day"   : Merge("LVL4Day",rootFolderOfDataOutputs)
              }
    
    
    
    
    
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round((end - start),0)) + " seconds")
