args <- commandArgs(trailingOnly = TRUE)
filteredDataInputCSVFileName <- args[1]
sensitivityAnalysisOutputCSVFileName <- args[2]

FILTERED_DATA <-read.csv(filteredDataInputCSVFileName, header=TRUE)
FILTERED_DATA <- FILTERED_DATA[,-1] # remove index column
EMULATOR_INPUT_DESIGN <- FILTERED_DATA[,1:ncol(FILTERED_DATA)-1]

OUTPUT_VECTOR <- FILTERED_DATA[,ncol(FILTERED_DATA)]

con <- file(filteredDataInputCSVFileName, "r")
firstLine <- readLines(con, n=1)
close(con)
strSplit <- strsplit(firstLine, ",")[[1]]
designVariableNames <- strSplit[3:length(strSplit)-1]

library(DiceKriging)

#MLE to do the sensitivity analysis only, a different method compared with emulator.

  EmModel<-km(formula=~.,design=EMULATOR_INPUT_DESIGN,response=OUTPUT_VECTOR,covtype="matern5_2",optim.method="BFGS",control=list(maxit=5000))
#####
##  - We set "maxit" to be large... This isn't necessary for a small example, but when you have a large number
##     of dimensions, convergence of the Gaussian process model parameter estimates can be slow, and so setting
##     this large means that it doesn't return a model that has not converged.
##  - The model fit here assumes that the underlying complex model is "deterministic", i.e. if you re-run the
##     same input combination, you will get the same output value.
##     => There are ways to account for noise in model outputs (using the input "noise.var", for example), but
##         I'm not going into this here... we can look at this later if it is needed.
##  - As the model is applied, R shows all sorts of model fit information. The main thing to note out of all
##     of this is that it has "converged" (at the bottom!...). It is also useful to check the number of
##     iterations used to obtain the Gaussian process parameter estimates: Every so often, the model can say
##     "converged" after only one iteration, and I am sceptical that the model has fitted properly when this
##     happens. If it only uses one iteration, I would re-fit (re-run the command).
##  - The resulting output, that we have named "EmModel", is a "km-object" in R...
#####
#####
## Initial checking of the model fit:
##  - The plot command used around a "km-object" will produce a set of generic "goodness-of-fit" plots for
##     the model - to give an initial idea of the model fit.
##  - For a good fit, the points in the "leave-one-out" plot at the top will follow closely to the line of
##     equality; the points in the standardised residual plot in the middle should show no real pattern
##     and lie in general between -2 and 2 on the y-axis scale; and the points in the Normal probability plot
##     at the bottom should follow closely to the diagonal line...
#####
  plot(EmModel)

library(sensitivity)

#####
##  We also need to define:
##   - A vector that lists the quantile functions of the distributions to be used for each parameter in the
##     sensitivity analysis. This is "DISTRIBUTION_VECTOR" in the call above. This vector will take pre-defined
##     distribution quantile functions in R as it's elements...
##   - A list object that provides the distribution parameters for each element of "DISTRIBUTION_VECTOR". This
##      is DISTRIBUTION_ARGUMENTS_LIST in the call above. This list object will define the parameters as they
##      are named in the pre-defined R distributions...
##   - The factors we are analysing sensitivity with respect to. This is just a list of the parameter names...
##   - "n" is the number of simulations per parameter that we want to use in the sensitivity call...
##      - This wants to be quite large...
##   - Finally, we will need any extra parameters that the "OUTPUT_FUNCTION" requires to run to be listed at
##      the end of the call. For us here, this means we will need to provide "m" - our fitted emulator model
##       (km-object).
## See example below for illustration of these bits!
#####
#####
## The output of fast99 has quite a few parts to it... all described in detail in the help file.
##  - By using the "plot()" command around the "fast99-object", we can get a plot of the sensitivity results...
##  - By typing the name of the "fast99-object", we see the variance contributions of the parameters.
##     => The "first order" column is what we refer to as "main effects", and are the main ones we look at...
##     However, the values R prints to screen are not direct outputs... and must be computed from other parts
##     of the object if we want to extract the actual numbers to use in R...
##  - To extract the "main effect" values (listed as "first order") - so the variance contributions from
##     individual parameters, we calculate:
##       "MainEffect<-SAout$D1/SAout$V"
##  - To extract the "total effect" values (listed as "total order") - so the total variance contribution for
##     each parameter, including all its interaction contributions, we calculate:
##       "TotalEffect<-1-(SAout$Dt/SAout$V)"
##  - To extract the "interaction variance contribution" for the parameters, we calculate:
##       "Interaction<-TotalEffect-MainEffect"
## **** The values of all of these are given as proportions... so to get these as percentage contributions, we
##       need to multiply up by 100.****
#####
#####
###############################################################################################################
#####
## EXAMPLE:
## So, for the 3-parameter cloud example here:
##  - We want to evaluate the parameter sensitivity, and variance contributions to the uncertainty in our model
##     output "acumR"...
##  - Define the "DISTRIBUTION_VECTOR" and "DISTRIBUTION_ARGUMENTS_LIST":
##    => As we have no prior information on the distributions of these parameters, I assume all values in our
##        parameter ranges are equally likely, and use the Uniform distribution for each parameter.
##        - The Uniform distribution has two parameters: "min" and "max", which I take from the defined
##           parameter ranges. (see: "help(qunif)" for more details on the Uniform Distribution functions in R)
##        - Note: As I fitted the emulator model with Input 3 on the log10-scale... the distribution used in the
##           sensitivity function must also be on this log10 scale...
#####
  DistVec <- rep("qunif", times=length(designVariableNames))
  DistArgsList <- list()
  for (ind in 1:length(designVariableNames)){
      DistArgsList[[ind]] <- list(min=min(EMULATOR_INPUT_DESIGN[,ind]),max=max(EMULATOR_INPUT_DESIGN[,ind]))
  }
#####
## - The parameter names can just be listed into a vector here... (They must be a vector of type "character")
## - We use "EMmean_fun" defined above as our "OUTPUT_FUNCTION". This must be copied into R.
## - I choose "n" to be 5000 - so in total, 3*5000=15000 simulations from the emulator will be used over the
##    uncertain parameter space to determine the parameter sensitivities (parameter uncertainty contributions)
##    to acumR.
## - Hence, the sensitivity function call is:
#####
EMmean_fun<-function(Xnew,m)  # emulator mean function
{
  predict.km(m,Xnew,"UK",se.compute=FALSE,checkNames=FALSE)$mean
}
library(DiceKriging)
library(sensitivity) ## library's commands don't need to be run every time we run the function... just the first time after starting R!
  SAout<-fast99(model=EMmean_fun,factors=designVariableNames,n=5000,q=DistVec,q.arg=DistArgsList,m=EmModel)
#####
## Display the output by typing the name of the "fast99-object":
#####
  SAout
#####
## Plot the sensitivity results:
#####
  #plot(SAout)

  #global sensitivity, sensitivity of the output, input values.  Note to the prior, but the input (prior for example) values
#####
## To extract the values for the sensitivity results to vectors:
##  - Can be useful if you wnat to make different plots etc!
#####
  MainEffect<-SAout$D1/SAout$V
  MainEffect
  TotalEffect<-1-(SAout$Dt/SAout$V)
  TotalEffect
  Interaction<-TotalEffect-MainEffect
  Interaction # ref the paper. globel sensitivity.

  sensitivityDataframe <- data.frame(designVariableNames, MainEffect, Interaction)
  write.csv(sensitivityDataframe, sensitivityAnalysisOutputCSVFileName, row.names=FALSE)

  barplot(MainEffect, main="LVL3Day Global variance-based sensitivity for wpos",
          xlab="MainEffect",names.arg = designVariableNames,cex.names = 0.65)
#####
#  lsq <- (c(2.5792,1.8482,0.3443,2.2652,2.2601,1.7981,1.5910,1.0245,2.5240))
#  lsq <- 1.0 / lsq
#  barplot(lsq, main="Lengthscala sensitivity for wpos",
#          xlab="Lengthscale",names.arg = designVariableNames,cex.names = 0.65)

###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
#####
