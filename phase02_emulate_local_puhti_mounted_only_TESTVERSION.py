#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:23:51 2021

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
print(__doc__)
import os
import time
from EmulatorData import EmulatorData

def main():

    rootFolderOfEmulatorSets = "/home/aholaj/mounttauskansiot/puhtiwork/eclair_training_simulations"
    rootFolderOfDataOutputs = "/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData"
    inputConfigFile = "/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/phase02.yaml"

    ###########
    emulatorSets = {"LVL3Night" : EmulatorData("LVL3Night",
                                             [rootFolderOfEmulatorSets,  "case_emulator_DESIGN_v3.0.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_night"],
                                             [rootFolderOfDataOutputs],
                                             inputConfigFile
                                             ),
                "LVL3Day"   :  EmulatorData( "LVL3Day",
                                            [rootFolderOfEmulatorSets, "case_emulator_DESIGN_v3.1.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_day"],
                                            [rootFolderOfDataOutputs],
                                            inputConfigFile
                                            ),
                "LVL4Night" :  EmulatorData("LVL4Night",
                                            [rootFolderOfEmulatorSets, "case_emulator_DESIGN_v3.2_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_night"],
                                            [rootFolderOfDataOutputs],
                                            inputConfigFile
                                            ),
                "LVL4Day"   : EmulatorData("LVL4Day",
                                            [rootFolderOfEmulatorSets, "case_emulator_DESIGN_v3.3_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_day"],
                                            [rootFolderOfDataOutputs],
                                            inputConfigFile
                                            )
                }

    emulatorSets["LVL3Night"].prepare()
    emulatorSets["LVL3Night"].runMethodAnalysis()
    emulatorSets["LVL3Night"].featureImportance()
    emulatorSets["LVL3Night"].bootStrap()
    emulatorSets["LVL3Night"].postProcess()
    aa = emulatorSets["LVL3Night"]
    
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nScript completed in { end - start : .1f} seconds")
