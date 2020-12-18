#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:41:12 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright

First phase of post-processing simulation data for a emulator
"""
print(__doc__)
import os
import pandas
import pathlib
import sys
import time


sys.path.append("../LES-03plotting")
from InputSimulation import InputSimulation
from Colorful import Colorful
from Data import Data

sys.path.append("../LES-emulator-01prepros")
import ECLAIR_calcs

from PostProcessingMetaData import PostProcessingMetaData

class Phase01(PostProcessingMetaData):
    def __init__(self, name : str, trainingSimulationRootFolder : list, dataOutputRootFolder : list):
        super().__init__(name, trainingSimulationRootFolder, dataOutputRootFolder)
        
        self.simulationDataCSVFileName = self.dataOutputRootFolder / self.name / (self.name + "_phase01.csv")
        
    def __checkIfOutputFileAlreadyExists(self):
        return self.simulationDataCSVFileName.is_file()
    
    def prepareEMULData(self):
        
        if (self.__checkIfOutputFileAlreadyExists()): print("Output file already exists"); return
        
        self.__prepareNewColumns()
        self.__getDesign()
        self.__joinDataFrames()
        self.__getPBLHinMeters()
        self.__setSimulationDataFrameAsJoined()
        self.__saveDataFrame()
        
    def __prepareNewColumns(self):
        self.fileList  = InputSimulation.getEmulatorFileList( self.trainingSimulationRootFolder )
        self.idList    = InputSimulation.getEmulatorIDlist( self.fileList )
        self.labelList = self.idList
        self.colorList = Colorful.getIndyColorList( len(self.fileList) )
        
        self.simulationData = InputSimulation( idCollection= self.idList, 
                                                   folderCollection= self.fileList,
                                                   labelCollection = self.labelList,
                                                   colorSet= self.colorList)

    def __getDesign(self):
        
        self.designData = InputSimulation.getEmulatorDesignAsDataFrame( self.trainingSimulationRootFolder, self.ID_prefix)

        
    def __joinDataFrames(self):        
        self.joinedDF = pandas.merge( self.simulationData.getSimulationDataFrame(), self.designData, on="ID")
        self.joinedDF = self.joinedDF.set_index("ID")
        
    def __getPBLHinMeters(self):
        pres0= 1017.8
        pblh_m_list  = [None]*self.joinedDF.shape[0]
        for ind,indValue in enumerate(self.joinedDF.index.values):
            tpot_pbl = self.joinedDF.loc[indValue]["tpot_pbl"]
            lwp = self.joinedDF.loc[indValue]["lwp"]
            pblh = self.joinedDF.loc[indValue]["pblh"]
            q_pbl      = ECLAIR_calcs.solve_rw_lwp( pres0*100., tpot_pbl,lwp*0.001, pblh*100. )  # kg/kg        
            lwp_apu, cloudbase, pblh_m, clw_max = ECLAIR_calcs.calc_lwp( pres0*100., tpot_pbl , pblh*100., q_pbl )
            pblh_m_list[ind] = pblh_m
        self.joinedDF["pblh_m"] = pblh_m_list
    
    def __setSimulationDataFrameAsJoined(self):
        
        self.simulationData.setSimulationDataFrame( self.joinedDF )
        
    def __saveDataFrame(self):
        self.simulationData.saveDataFrameAsCSV(self.simulationDataCSVFileName)
    
def main():
    
    rootFolderOfEmulatorSets = os.environ["EMULATORDATAROOTFOLDER"]
    rootFolderOfDataOutputs = os.environ["EMULATORPOSTPROSDATAROOTFOLDER"]
    
    emulatorSets = {"LVL3Night" : Phase01("LVL3Night",
                                             [rootFolderOfEmulatorSets,  "case_emulator_DESIGN_v3.0.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_night"],
                                             [rootFolderOfDataOutputs],
                                             ),
                "LVL3Day"   :  Phase01( "LVL3Day",
                                            [rootFolderOfEmulatorSets, "case_emulator_DESIGN_v3.1.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_day"],
                                            [rootFolderOfDataOutputs],
                                            ),
                "LVL4Night" :  Phase01("LVL4Night",
                                            [rootFolderOfEmulatorSets, "case_emulator_DESIGN_v3.2_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_night"],
                                            [rootFolderOfDataOutputs],
                                            ),
                "LVL4Day"   : Phase01("LVL4Day",
                                            [rootFolderOfEmulatorSets, "case_emulator_DESIGN_v3.3_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_day"],
                                            [rootFolderOfDataOutputs],
                                            )
                }
    
    for key in emulatorSets:
        emulatorSets[key].prepareEMULData()
    
    
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Script completed in {Data.timeDuration(end - start):s}")
