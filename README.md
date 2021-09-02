# LES-emulator-02postpros

This repository holds scripts to create parameterisations for Ahola et al 2021.

## Instructions for use

### Download LES data

Data from UCLALES-SALSA runs is available from (DOI:10.23728/fmi-b2share.296483f247b1412ebd27f0b82dd1bb76)[https://fmi.b2share.csc.fi/records/296483f247b1412ebd27f0b82dd1bb76].

Only following subfolders are needed to reproduce the results of Ahola et al 2021
- case_emulator_DESIGN_v3.0.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_night  
(shortname **LVL3Night**, contains SB nighttime simulations)
- case_emulator_DESIGN_v3.1.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_day  
(shortname **LVL3Day**, contains SB daytime simulations)
- case_emulator_DESIGN_v3.2_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_night  
(shortname **LVL4Night**, contains SALSA nighttime simulations)
- case_emulator_DESIGN_v3.3_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_day  
(shortname **LVL4Day**, contains SALSA daytime simulations)

### Configuration files

- Download configuration files from (LES-emulator-04configFiles)[https://github.com/JaakkoAhola/LES-emulator-04configFiles].
	- locationsMounted.yaml
	- phase02.yaml

- Modify locationsMounted.yaml according to the instructions at (LES-emulator-04configFiles)[https://github.com/JaakkoAhola/LES-emulator-04configFiles]

- Use condaEnvironment.yaml as a base for python conda environment.

### environment variables

Set environment variable LESMAINSCRIPTS to point to the location of the library (LES-03plotting)[https://github.com/JaakkoAhola/LES-03plotting].

Set environment variable PYTHONEMULATOR to point to the location of the library (GPEmulatorPython)[https://github.com/JaakkoAhola/GPEmulatorPython].

### running (phase01)[phase01_prepareEMULforPlotting.py]

run (phase01_prepareEMULforPlotting.py)[phase01_prepareEMULforPlotting.py] with a command  
`python phase01_prepareEMULforPlotting.py locationsMounted.yaml`

### running (phase02)[phase02_emulate_local_puhti_mounted.py]

run (phase02_emulate_local_puhti_mounted.py)[phase02_emulate_local_puhti_mounted.py] with a command  
`python phase02_emulate_local_puhti_mounted.py locationsMounted.yaml`


## Important notes

The main output is {simulationSet}_complete.csv in a subfolder that will be in `postProsDataRootFolder` indicated by locationsMounted.yaml. Here {simulationSet} is the shortname of the simulation set.


(EmulatorData.py)[EmulatorData.py] contains the main methods in developing the parameterisations along with the GPE library (GPEmulatorPython)[https://github.com/JaakkoAhola/GPEmulatorPython].


## Authors

Jaakko Ahola,  Tomi Raatikainen, Antti Kukkurainen, Muzaffer Ege Alper, Jia Liu, Jukka-Pekka Keskinen, Antti Lipponen

### Author contributions

JA made the LF results with help from TR.
JA made the LFRF results with development help from AL and AK.
JA made the GPE results with development help from MEA, AK, JL, JPK and AL.
JA wrote the python code with help from AK and AL.
