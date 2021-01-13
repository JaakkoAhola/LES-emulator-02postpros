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


sys.path.append(os.environ["LESMAINSCRIPTS"])
from InputSimulation import InputSimulation
from Colorful import Colorful
from Data import Data

sys.path.append("../LES-emulator-01prepros")
import ECLAIR_calcs

from PostProcessingMetaData import PostProcessingMetaData

class Phase01(PostProcessingMetaData):

    def prepareEMULData(self):

        if (self.__checkIfOutputFileAlreadyExists()): print("Output file already exists"); return

        self.__prepareNewColumns()
        self.__getDesign()
        self.__joinDataFrames()
        self.__getPBLHinMetersColumn()
        self.__setSimulationDataFrameAsJoined()
        self.__saveDataFrame()

    def __checkIfOutputFileAlreadyExists(self):
        return self.phase01CSVFile.is_file()

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
        self.simulationDataFrame = pandas.merge( self.simulationData.getSimulationDataFrame(), self.designData, on="ID")
        self.simulationDataFrame.set_index("ID", inplace = True)

    def __getPBLHinMetersColumn(self):
        self.simulationDataFrame["pblh_m"] = self.simulationDataFrame.apply(lambda row: \
                                                          Phase01.getPBLHinMeters( {"pres0": 1017.8,
                                                                                    "tpot_pbl":row["tpot_pbl"],
                                                                                    "lwp": row["lwp"],
                                                                                    "pblh": row["pblh"]} ),
                                                              axis = 1)

    def getPBLHinMeters( parametersDict = {"pres0": None, "tpot_pbl":None, "lwp": None, "pblh":None} ):
        """
        Parameters
        ----------
        parametersDict : dict,
            DESCRIPTION. The default is {"pres0": None, "tpot_pbl":None, "lwp": None, "pblh":None}.

            Units:
                pres0 : [hPa]
                tpot_pbl : [K]
                lwp : [g/kg]
                pblh : [hPa]

        Returns
        -------
        pblh_m : unit [m]
            DESCRIPTION: Planetary Boundary Layer Height in meters

        """

        for key in ["pres0", "tpot_pbl", "lwp", "pblh"]:
            assert(key in parametersDict)

        q_pbl = ECLAIR_calcs.solve_rw_lwp(parametersDict["pres0"]*100.,
                                          parametersDict["tpot_pbl"],
                                          parametersDict["lwp"]*0.001,
                                          parametersDict["pblh"]*100.) #kg/kg

        lwp_apu, cloudbase, pblh_m, clw_max = ECLAIR_calcs.calc_lwp( parametersDict["pres0"]*100.,
                                                                    parametersDict["tpot_pbl"],
                                                                    parametersDict["pblh"]*100.,
                                                                    q_pbl )

        return pblh_m

    def __setSimulationDataFrameAsJoined(self):

        self.simulationData.setSimulationDataFrame( self.simulationDataFrame )

    def __saveDataFrame(self):
        self.simulationData.saveDataFrameAsCSV(self.phase01CSVFile)

    def removeOutputFile(self):
        os.remove(self.phase01CSVFile)

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
