#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:23:51 2021

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
print(__doc__)
import time
import sys
from EmulatorData import EmulatorData

def main(inputConfigFile):

    rootFolderOfEmulatorSets = "/fmi/scratch/project_2001927/aholaj/eclair_training_simulations"
    rootFolderOfDataOutputs = "/fmi/scratch/project_2001927/aholaj/EmulatorManuscriptData"

    ###########
    emulatorSets = {"LVL3Day" : EmulatorData("LVL3Day",
                                             [rootFolderOfEmulatorSets,  "case_emulator_DESIGN_v3.1.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_day"],#/fmi/scratch/project_2001927/aholaj/eclair_training_simulations/case_emulator_DESIGN_v3.1.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_day
                                             [rootFolderOfDataOutputs],
                                             inputConfigFile
                                             )
                }

    for key in emulatorSets:
        emulatorSets[key].prepare()
        emulatorSets[key].runEmulator()
        emulatorSets[key].postProcess()
if __name__ == "__main__":
    start = time.time()
    try:
        inputConfigFile = sys.argv[1]
    except IndexError:
        inputConfigFile = "/fmi/scratch/project_2001927/aholaj/EmulatorManuscriptData/phase02.yaml"
    print("inputConfigFile", inputConfigFile)

    main(inputConfigFile)

    end = time.time()
    print(f"\nScript completed in { end - start : .1f} seconds")
