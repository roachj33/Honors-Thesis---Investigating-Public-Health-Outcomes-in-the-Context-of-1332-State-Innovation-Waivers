# Honors-Thesis---Investigating-Public-Health-Outcomes-in-the-Context-of-1332-State-Innovation-Waivers
This repository contains all resources and reports from my Economics Honors Program thesis, "Investigating Public Health Outcomes in the Context of 1332 State Innovation Waivers", an exploration of the causal impacts of state-based reinsurance programs on the health and health-related behaviors of statewide populations. Further and more detailed explanations of the methodology of this research are present within the pdf upload of the thesis, but, at its core, this project focused on applying a Difference-in-Differences research design to data from the Behavioral Risk Factor Surveillance System (BRFSS) over the course of 2012 to 2019 to ascertain the causal effects of a state implementing a reinsurance program under a 1332 waiver on the health of that state's citizens. More general information about the BRFSS can be located [here](https://www.cdc.gov/brfss/about/index.htm). The data sets used in this research can be downloaded from [this link](https://www.cdc.gov/brfss/annual_data/annual_data.htm) by downloading the SAS zip files from each sub-page, beginning in 2012 up until 2019, unzipping those files and storing the correspond "LLCP" data files in the same working directory as that of the coding files. The code written to complete this research was done in both python and R in the following ways:

* Python was utilized to read in, re-code, clean and merge together the desired data set for analysis, as well as to conduct preliminary weighting calculations and test out the validity of our data in satisfying the assumptions of our model. This work was done in the file `simplify.py` while `test.py` provides testing code for the functions contained within the former module. It is expected that the user has these packages for those python files to run properly:
  * os
  * pandas
  * glob
  * numpy
  * plotly.express
  * pandas.testing
* R was deployed for regression analysis on all models tested within the thesis. This thesis made use of linear regression models within a difference-in-differences framework both with and without subject matching. Moreover, as a measure of sensitivity analysis, a separate .Rmd file re-performs the methodologies on a subset of states which strictly adopted medicaid expansions, alleviating a potential confounder cited in the paper itself, since one state (Wisconsin) which implemented a reinsurance program under these waivers did not adopt any such policies; the difference between these two analyses is conveyed within their respective file names. An additional 3rd .Rmd file was utilized for the creation of the tables found in the "Results" section of the thesis. For the .Rmd files to run properly, the user must have installed the packages:
  * ggplot2
  * faraway
  * MatchIt
  * lmtest
  * sandwich
  * survey
  * glmnet
  * caret
  * mlogit
  * cobalt
  * WeightIt

All primary analysis reported on within this thesis adheres strictly to BRFSS standards for weighting and preparation for analysis of this survey data, as found by accessing the pdf "Complex Sampling Weights and Preparing Module Data for Analysis CDC" for any given year's annual survey data, which can be located via the second hyperlink provided in this `README` file. Any qualifications to these findings to this thesis are presented and addressed within the paper itself.
