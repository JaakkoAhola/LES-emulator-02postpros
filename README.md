# Postprosessing data for emulator

## Set environment variables

- set `$EMULATORDATAROOTFOLDER` for folder where all the LES training data sets are stored. This directory should contain folders:
	- case_emulator_DESIGN_v3.0.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_night (Short name of training set: LVL3Night)
	- case_emulator_DESIGN_v3.1.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_day (Short name of training set: LVL3Day)
	- case_emulator_DESIGN_v3.2_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_night (Short name of training set: LVL4Night)
	- case_emulator_DESIGN_v3.3_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_day (Short name of trainingn set: LVL4Day)

- set `$EMULATORPOSTPROSDATAROOTFOLDER` for folder where you want to store your postprosessing data that the scripts produce

- set `$GPTRAINEMULATOR` that contains executable: `gp_train` obtained from [GPEmulator repository](https://github.com/JaakkoAhola/GPEmulator/programs/emulator_train)
- set `$GPPREDICTEMULATOR` that contains executable: `gp_predict` obtained from [GPEmulator repository](https://github.com/JaakkoAhola/GPEmulator/programs/emulator_predict)


## Prerequisites

- (getSensitivityAnalysis.bash)[getSensitivityAnalysis.bash]
	- library (DiceKriging v1.5.6)[https://www.rdocumentation.org/packages/DiceKriging/versions/1.5.6] installed with commands:
		- `sudo R`
		- install.packages("https://github.com/DiceKrigingClub/DiceKriging/releases/download/osx-linux/DiceKriging_1.5.6.tar.gz")
	- library (sensitivity v1.21.0)[https://www.rdocumentation.org/packages/sensitivity/versions/1.21.0] installed with commands:
		- `sudo R`
		- install.packages("sensitivity")



## Description of scripts and their outputs

Here {TrainingSet} is the short name of the training set, i.e. one of LVL3Day, LVL3Night, LVL4Day or LVL4Night.

- (getEmulatorData.py)[getEmulatorData.py] produces the following files within `$EMULATORPOSTPROSDATAROOTFOLDER/{TrainingSet}` :

	- The design of training simulations, based on Binary Space Partitioning method. Here copied from training data folder and renamed as: {TrainingSet}_design.csv
	- Filtered data in FORTRAN friendly format: {TrainingSet}_DATA
	- Filtered data in csv-format: {TrainingSet}_filtered.csv
	- Response of UCLALES-SALSA training simulations. In this case the updraft velocity at cloud base: {TrainingSet}_wpos.csv
	- subfolder DATA consists of omitting one point from training data and predicting that point with emulator
	- Simulated vs Emulated data: *TODO*

- (getSensitivityAnalysis.bash)[getSensitivityAnalysis.bash] calls R-script (sensitivityAnalysis.r)[sensitivityAnalysis.r] for each training set

	- (sensitivityAnalysis.r)[sensitivityAnalysis.r] produces the following sensitivity analysis results:
		- {TrainingSet}_sensitivityAnalysis.csv
