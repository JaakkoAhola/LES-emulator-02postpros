#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:23:51 2021

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
import os
import time
import sys
from EmulatorData import EmulatorData
sys.path.append(os.environ["LESMAINSCRIPTS"])
from Data import Data


def main():

    try:
        locationsFile = sys.argv[1]
    except IndexError:
        locationsFile = "/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData_revision_v0.1/locationsMounted.yaml"

    ###########
    emulatorSets = {"LVL3Night": EmulatorData("LVL3Night",
                                              "case_emulator_DESIGN_v3.0.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_night",
                                              locationsFile
                                             ),
                    "LVL3Day": EmulatorData("LVL3Day",
                                            "case_emulator_DESIGN_v3.1.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_day",
                                            locationsFile
                                            ),
                    "LVL4Night": EmulatorData("LVL4Night",
                                              "case_emulator_DESIGN_v3.2_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_night",
                                              locationsFile
                                              ),
                    "LVL4Day": EmulatorData("LVL4Day",
                                            "case_emulator_DESIGN_v3.3_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_day",
                                            locationsFile
                                            )
                    }

    for key in emulatorSets:
        emulatorSets[key].prepare()
        emulatorSets[key].runMethodAnalysis()
        emulatorSets[key].featureImportance()
        # emulatorSets[key].bootStrap()
        emulatorSets[key].postProcess()

if __name__ == "__main__":
    start = time.time()
    now = datetime.now().strftime('%d.%m.%Y %H.%M')
    print(f"Script started {now}.")
    main()
    end = time.time()
    now = datetime.now().strftime('%d.%m.%Y %H.%M')
    print(f"Script completed {now} in {Data.timeDuration(end - start)}")
