"""
Name: Joey Roach
Date: May 17th, 2021

simplify.py is a python script which cleanses the BRFSS survey data from 2012
to 2019 by merging together all annual datasets and writing the joined result
as a new CSV file. This script assumes that all BRFSS data files required for
the analysis have already been downloaded and are located individually within
the current working directory. The script further cleans the CSV file by
recoding ambiguous survey responses (i.e. "I do not know",
"I do not want to answer", etc) into non-answers and filtering those responses
out. After recoding and dropping missing values, the survey weights are
re-weighted according to BRFSS documentation standards and constructs the
indicator variables required for the difference-in-differences research design
outlined in the thesis. Additionally, it performs recoding of multiple response
levels into binary format for simplification of the DiD model, and computes
proportional outcomes as defined in the thesis as well as parallel trends
visualizations, which are also saved in a sub-folder entitled
'parallel_trends_graphs', which this module creates if no such directory
currently exists on the user system.
"""


import os
import pandas as pd
import glob
import numpy as np
import plotly.express as px


def rewrite_data(current_wd):
    """
    Takes BRFSS XPT data from 2012 to 2019 and simplifies down to variables
    of interest for research, renames variables across years for consistency
    and writes each yearly data file to a csv format file. Stores these files
    in the same current working directory.

    Arguments:
        current_wd: A string representing the current working directory.

    Returns:
        None.
    """
    # Collect all annual BRFSS data file names.
    all_files = glob.glob(current_wd + '/LLCP201[2-9]*')
    for year in all_files:
        annual_data = pd.read_sas(year)
        # These "base" variables are ones whose names are consistent across all
        # years.
        my_vars = ['_STATE', 'IYEAR', 'NUMADULT', 'GENHLTH', 'PHYSHLTH',
                   'MENTHLTH', 'POORHLTH', 'HLTHPLN1', 'PERSDOC2', 'MEDCOST',
                   'CHECKUP1', 'CVDCRHD4', 'CVDINFR4', 'CVDSTRK3', 'ASTHNOW',
                   'CHCSCNCR', 'CHCOCNCR', 'MARITAL', 'EDUCA',
                   'RENTHOM1', 'CHILDREN', 'INCOME2', 'HIVTSTD3', '_AGE65YR',
                   '_RFBMI5', '_LLCPWT', '_STSTR']
        # These variables are all the same except for the year 2019, so we can
        # use them as well to help for all years before 2019.
        mostly_same = ['CHCCOPD1', 'HAVARTH3', 'ADDEPEV2', 'DIABETE3',
                       'FLSHTMY2', 'HIVTST6']
        # Grab specific variables that need to be renamed for consistency
        # from each year, then limit data down to columns of interest and
        # rename as needed.
        if year == (current_wd + '\\LLCP2012.XPT'):
            additional = ['CHCKIDNY', 'SEX', 'EMPLOY', '_RACE_G']
            my_vars.extend(additional)
            my_vars.extend(mostly_same)
            concise = annual_data[my_vars]
            concise = concise.rename(columns={'EMPLOY': 'EMPLOY1',
                                     '_RACE_G': '_RACE_G1'})
        elif year == (current_wd + '\\LLCP2018.XPT'):
            additional = ['CHCKDNY1', 'SEX1', 'EMPLOY1', '_RACE_G1']
            my_vars.extend(additional)
            my_vars.extend(mostly_same)
            concise = annual_data[my_vars]
            concise = concise.rename(columns={'CHCKDNY1': 'CHCKIDNY',
                                              'SEX1': 'SEX'})
        elif year == (current_wd + '\\LLCP2019.XPT'):
            additional = ['CHCCOPD2', 'HAVARTH4', 'CHCKDNY2', 'DIABETE4',
                          'SEXVAR', 'EMPLOY1', 'FLSHTMY3', 'HIVTST7',
                          '_RACE_G1', 'ADDEPEV3']
            my_vars.extend(additional)
            concise = annual_data[my_vars]
            concise = concise.rename(columns={'CHCCOPD2': 'CHCCOPD1',
                                              'HAVARTH4': 'HAVARTH3',
                                              'ADDEPEV3': 'ADDEPEV2',
                                              'CHCKDNY2': 'CHCKIDNY',
                                              'DIABETE4': 'DIABETE3',
                                              'SEXVAR': 'SEX',
                                              'FLSHTMY3': 'FLSHTMY2',
                                              'HIVTST7': 'HIVTST6'})
        else:
            additional = ['CHCKIDNY', 'SEX', 'EMPLOY1', '_RACE_G1']
            my_vars.extend(additional)
            my_vars.extend(mostly_same)
            concise = annual_data[my_vars]
        # Rename some cumbersome column names to more convenient ones.
        annual_data = annual_data.rename(columns={'_STATE': 'STATE',
                                                  'IYEAR': 'YEAR',
                                                  'HLTHPLN1': 'HLTHPLN',
                                                  'PERSDOC2': 'PERSDOC',
                                                  'CHECKUP1': 'CHECKUP',
                                                  'CVDCRHD4': 'HRTDIS',
                                                  'CVDINFR4': 'HRTATTCK',
                                                  'CVDSTRK3': 'STROKE',
                                                  'CHCSCNCR': 'SKNCNCR',
                                                  'CHCOCNCR': 'OTHERCNCR',
                                                  'RENTHOM1': 'RENT',
                                                  'INCOME2': 'INCOME',
                                                  '_AGE65YR': 'AGE',
                                                  '_RFBMI5': 'BMI',
                                                  '_LLCPWT': 'WEIGHT',
                                                  'CHCCOPD1': 'COPD',
                                                  'HAVARTH3': 'ARTH',
                                                  'CHCKIDNY': 'KIDDIS',
                                                  'DIABETE3': 'DIABETE',
                                                  'EMPLOY1': 'EMPLOY',
                                                  '_RACE_G1': 'RACE',
                                                  'ADDEPEV2': 'DEPRESS',
                                                  '_STSTR': 'STSTR'})
        # Get year of current data file, convert it to a csv file, including
        # which year the csv file represents.
        annual_digit = year[-5]
        concise.to_csv(current_wd + '/brfss201' + annual_digit + '.csv')


def check_rewrite(current_wd):
    """
    Function to check that columns are consistent across all years, and
    that each year contains the number of observations that it should, as
    per the BRFSS documentation.

    Arguments:
        current_wd: A string representing the current working directory.

    Returns:
        A list of 8 2-tuples, each containing the set of brfss column names
        in the first entry, and the number of observations of the data set in
        the second.
    """
    # Identify all cleaned BRFSS csv files.
    all_csvs = glob.glob(current_wd + '/brfss201[2-9]*')
    to_return = []
    for csv in all_csvs:
        # For each annual BRFSS file, grab column names and number of rows
        # and store the results.
        test_data = pd.read_csv(csv)
        col_names = set(test_data.columns)
        data_size = len(test_data)
        entry = (col_names, data_size)
        to_return.append(entry)
    return to_return


def recode_data(annual_data):
    """
    Given a data frame of one years' worth of BRFSS data, accounts for
    all variables in the data that contained potential responses of either
    "I don't know" or a refusal to give an answer. Actual integer encoding
    of these variables depends upon the particular question. Such responses
    are converted to NA's, as per the standard when working with BRFSS data.
    Returns the data frame once those responses have been converted. This is a
    helper function used within the clean_data method.

    Arguments:
        annual_data: A data frame of one year of BRFSS data.

    Returns:
        A pandas data frame with the above responses replaced with NA values.
    """
    # Mostly two types of encodings for these responses: 77/99, or 7/9,
    # corresponding to "I don't know" and "I don't want to answer",
    # respectively.
    # Break all variables that adhere to this structure and need to be
    # recoded up into the appropriate group.
    # First, those variables with 7 and 9's for those responses.
    vars_one = ['GENHLTH', 'HLTHPLN', 'PERSDOC', 'MEDCOST', 'CHECKUP',
                'HRTDIS', 'HRTATTCK', 'STROKE', 'SKNCNCR', 'OTHERCNCR',
                'COPD', 'ARTH', 'KIDDIS', 'DIABETE', 'DEPRESS', 'MARITAL',
                'EDUCA', 'RENT', 'BMI', 'SEX']
    # Get variables with responses of the 77/99 variety now.
    vars_two = ['PHYSHLTH', 'MENTHLTH', 'POORHLTH', 'INCOME']
    # Now deal with outlier variables which use alternative encoding on their
    # own.
    # 7 has a different encoding for Employ, do not replace it.
    annual_data['EMPLOY'] = annual_data['EMPLOY'].replace(9, np.nan)
    # 77 (potentially) has a meaningful encoding for children, do not replace.
    annual_data['CHILDREN'] = annual_data['CHILDREN'].replace(99, np.nan)
    # 3 corresponds to non-responses in the variable age, so replace it.
    annual_data['AGE'] = annual_data['AGE'].replace(3, np.nan)
    # The following variables also utilize the encoding of '88' to signify
    # the response 'None', so re-code those as well.
    vars_88 = vars_two[0:3] + ['CHILDREN']
    for var in vars_88:
        annual_data[var] = annual_data[var].replace(88, 0).astype('int64')
    # Now to remove responses for the more common encodings.
    to_remove_one = [7, 9]
    to_remove_two = [77, 99]
    # Link together variables with their respective encodings, and re-code
    # each response with one of those encodings as NaN for every variable.
    all_vars = [(vars_one, to_remove_one), (vars_two, to_remove_two)]
    for group in all_vars:
        for var in group[0]:
            for code in group[1]:
                annual_data[var] = annual_data[var].replace(code, np.nan)
    return annual_data


def clean_data(current_wd):
    """
    For each csv of annual BRFSS data, drop columns that most cause
    missingness, convert year variable to an integer, rename columns
    and drops outright NA's. Returns a list where each entry is one years'
    worth of cleansed BRFSS data.

    Arguments:
        current_wd: The current working directory for simplify.py, which
                    contains each yearly BRFSS csv file.

    Returns:
        A list of 8 entries, each corresponded to a cleaned annual BRFSS
        pandas dataframe.
    """
    all_csvs = glob.glob(current_wd + '/brfss201[2-9]*')
    cleaned_data = []
    for csv in all_csvs:
        annual_data = pd.read_csv(csv)
        # Drop columns that contribute most to missingness.
        annual_data = annual_data.drop(columns=['Unnamed: 0', 'HIVTST6',
                                                'FLSHTMY2', 'HIVTSTD3',
                                                'ASTHNOW'])
        # Convert IYEAR variable from object to int type.
        remove = [('b', ''), ("'", '')]
        for one, two in remove:
            annual_data['IYEAR'] = annual_data['IYEAR'].str.replace(one, two)
        annual_data['IYEAR'] = annual_data['IYEAR'].str.strip().astype('int64')
        annual_data['IYEAR'] = annual_data['IYEAR'].astype('int64')
        # Recode variables that need to be recoded.
        annual_data = recode_data(annual_data)
        # Drop missing values from data.
        annual_data = annual_data.dropna()
        cleaned_data.append(annual_data)
    return cleaned_data


def acquire_state_data(all_data, state_id):
    """
    Given a list of all_data whose entries are cleaned BRFSS survey dataframes,
    as in the form returned by the clean_data function, and a state identifier
    code, pulls all brfss data for that state, returning a list of 8 entries,
    where each entry represents that given state's annual BRFSS data.

    Arguments:
        all_data: A list where each entry represents a years' worth of BRFSS
                  data in a pandas dataframe.
        state_id: an integer state identifier, corresponding to BRFSS FIPS
                  codes.

    Returns:
        A list where each entry is a years' worth of BRFSS data for one state.
    """
    all_state_data = []
    for year in all_data:
        # Filter down for only data of the desired state.
        mask = year['STATE'] == state_id
        data = year[mask]
        all_state_data.append(data)
    return all_state_data


def get_state_obs(state_data):
    """
    Given a list of pandas data frames, calculates the number of observations
    present in each data frame, and returns the total number of observations
    present in the list. This is a helper function implemented in the
    weight_state_data method.

    Arguments:
        state_data: a list of pandas data frames of annual BRFSS data (intended
                    to be state-level data, like the form returned by
                    acquire_state_data).

    Returns:
        An integer representing the total number of observations in the list.
    """
    total_obs = 0
    for dataset in state_data:
        # Calculate number of observations in yearly dataset, update running
        # total number of observations.
        yearly_obs = len(dataset)
        total_obs += yearly_obs
    return total_obs


def weight_state_data(state_data):
    """
    Given a list where each entry is a pandas data frame representing a year's
    worth of state BRFSS data, reweights the weight variable according to
    BRFSS standards. That is, the weight of observations within a given year
    is reweighted, where the reweight is defined as the original weight
    multiplied by the proportion of observations within that year relative
    to all observations for that state within the dataset.

    Arguments:
        state_data: a list where each entry is a pandas dataframe that
                    represents a year's worth of one state's BRFSS data,
                    as returned by acquire_state_data.

    Returns:
        A list of the same format as the input, but with a reweight variable
        added on as a column.
    """
    # Obtain total number of observations for given state, from 2012-2019.
    total_obs = get_state_obs(state_data)
    reweighted_data = []
    for year in state_data:
        reweight = year.copy()
        # Determine number of observations within given year.
        yearly_obs = len(reweight)
        # Calculate proportion of given year's observations to total number of
        # observations.
        weight_prop = yearly_obs / total_obs
        # Define the reweight according to BRFSS data documentation procedure.
        reweight['REWEIGHT'] = reweight['WEIGHT'] * weight_prop
        reweighted_data.append(reweight)
    return reweighted_data


def consolidate_data(datasets):
    """
    Given a list where each entry is a pandas data frame, concatenates each
    element of the list into one pandas dataframe, returning the merged one.

    Arguments:
        datasets: A list where each entry is a pandas dataframe. Designed to
                  work on both input lists with entires of individual state
                  annual data (i.e., those of the form returned by
                  acquire_state data) and those where each entry represents
                  an entire state's BRFSS data from 2012-2019.

    Returns:
        A pandas dataframe with each list entry concatenated together.
    """
    total_data = pd.concat(datasets, ignore_index=True)
    return total_data


def convert_int_type(column_series):
    """
    Given a pandas series of type numpy.int32, converts the series to type
    int64 and returns that converted series. This is a helper function to
    the method contruct_additional_vars.

    Arguments:
        column_series: A pandas series with type numpy.int32.

    Returns:
        A pandas series identical to the input, except with each column's type
        converted to int64.
    """
    converted = column_series.astype('int64')
    return converted


def construct_additional_vars(all_data):
    """
    Given the complete set of all available BRFSS data from 2012-2019 for all
    states that we want to analyze in our research, creates the appropriate
    treatment, time and disease indicator variables intended for later DiD
    regression analysis, working on the basis of multiple time periods.

    Arguments:
        all_data: A pandas dataframe containing all available, usable BRFSS
                  data from 2012-2019 in both treatment and control states.

    Returns:
        A pandas dataframe which mirrors the input one, with the addition
        of treatment, time and disease indicator variables.
    """
    # Early treatment state conditions.
    early_treat = (all_data['STATE'] == 2) | (all_data['STATE'] == 27) | \
                  (all_data['STATE'] == 41)
    # Time conditions for early treatment states.
    early_treat_time = all_data['YEAR'] >= 2018
    # Late treatment state conditions.
    late_treat = (all_data['STATE'] == 23) | (all_data['STATE'] == 24) | \
                 (all_data['STATE'] == 55)
    # Time conditions for late treatment states.
    late_treat_time = all_data['YEAR'] >= 2019
    # Identify all treatment states with a 1, control states with a 0.
    all_data['TREAT_EARLY'] = np.where(early_treat, 1, 0)
    all_data['TREAT_LATE'] = np.where(late_treat, 1, 0)
    # Identify all years in during or after which reinsurance
    # was implemented with a 1, all other instances (i.e., control states or
    # treatment states prior to the year of implementation) with a 0.
    all_data['TIME_EARLY'] = np.where(early_treat_time, 1, 0)
    all_data['TIME_LATE'] = np.where(late_treat_time, 1, 0)
    # Chronic disease conditions
    dis_conds = (all_data['HRTDIS'] == 1) | (all_data['HRTATTCK'] == 1) | \
                (all_data['STROKE'] == 1) | (all_data['SKNCNCR'] == 1) | \
                (all_data['OTHERCNCR'] == 1) | (all_data['BMI'] == 2) | \
                (all_data['COPD'] == 1) | (all_data['ARTH'] == 1) | \
                (all_data['KIDDIS'] == 1) | (all_data['DIABETE'] == 1) | \
                (all_data['DIABETE'] == 2) | (all_data['DEPRESS'] == 1)
    # Assign those with a chronic disease a 1, those with none a 0.
    all_data['CHR_DIS'] = np.where(dis_conds, 1, 0)
    # Convert introduced variables of type Numpy int32 type to that of int64.
    to_convert = ['TREAT_EARLY', 'TREAT_LATE', 'TIME_EARLY', 'TIME_LATE',
                  'CHR_DIS']
    for conversion in to_convert:
        all_data[conversion] = convert_int_type(all_data[conversion])
    return all_data


def binary_recode(total_data):
    """
    Given the complete_data.csv file in a pandas data frame, recodes selected
    variables to match a binary format in the following ways: For the variables
    HLTHPLN, MEDCOST, HRTDIS, HRTATTCK, STROKE, SKNCNCR, OTHERCNCR, KIDDIS,
    COPD, ARTH, and DEPRESS converts '2.0' to '0.0', where a 2.0 means "no"
    in the survey, and we are only switching to the more traditional binary
    format. The variable BMI currently has a '1' for no and a '2' for yes,
    so switch '2' to a '1' and '1' to a '0'. Also, converts these and any
    other variables in float format to integer format for simplifying purposes.

    Arguments:
        total_data: A pandas data frame of the complete_data.csv.

    Returns:
        An updated pandas data frame with the alterations mentioned above.
    """
    # Variables to convert 2's to 0's to match up with standard binary format.
    # Also, changes these variables to integer format.
    variables_binary = ['HLTHPLN', 'MEDCOST', 'HRTDIS', 'HRTATTCK', 'STROKE',
                        'SKNCNCR', 'OTHERCNCR', 'KIDDIS', 'COPD', 'ARTH',
                        'DEPRESS']
    for variable in variables_binary:
        total_data[variable] = \
            total_data[variable].replace(2, 0).astype('int64')
    # BMI is slightly different so deal with it independently.
    total_data['BMI'] = total_data['BMI'].replace(1, 0)
    total_data['BMI'] = total_data['BMI'].replace(2, 1).astype('int64')
    # Convert remaining float variables to integer format, except for WEIGHT
    # and REWEIGHT.
    other_vars = ['STATE', 'NUMADULT', 'GENHLTH', 'PERSDOC', 'CHECKUP',
                  'MARITAL', 'EDUCA', 'RENT', 'INCOME', 'AGE', 'SEX',
                  'EMPLOY', 'RACE', 'DIABETE']
    for variable in other_vars:
        total_data[variable] = total_data[variable].astype('int64')
    return total_data


def proportions_prep_for_outcomes(recoded_data):
    """
    Given a recoded_data pandas data frame of the type returned by
    binary_recode, simplifies outcome variables down to binary
    status to prepare them for usage in terms of proportions. Does
    so by adding the following variables, retaining the variables
    with which they are based off of, in the following ways:
        'GOOD_GENHLTH': Encoded as a 1 for responses to GENHLTH that
                        are a 1, 2, or 3; else is 0.
        'PHYS_DISTRESS': Encoded as a 1 for responses to PHYSHLTH that
                         are 14 or greater; else is 0.
        'MENT_DISTRESS': Encoded as a 1 for responses to MENTHLTH that
                         are 14 or greater; else is 0.
        'POOR_OVR_HLTH': Encoded as a 1 for responses to POORHLTH that
                         are 14 or greater; else is 0.
        'HAS_PERSDOC': Encoded as a 1 for responses to PERSDOC that are
                       either a 1 or 2; else is 0.
        'ANNUAL_CHECKUP': Encoded as a 1 for responses to CHECKUP that are
                          a 1; else is 0.

    Arguments:
        recoded_data: A pandas data frame of the format returned by
                      recode_variables.

    Returns:
        A pandas data frame with the above described variables added.
    """
    # Get condition checks for each binary outcome variable.
    genhlth_cond = (recoded_data['GENHLTH'] == 4) | \
        (recoded_data['GENHLTH'] == 5)
    phys_distress_cond = recoded_data['PHYSHLTH'] >= 14
    ment_distress_cond = recoded_data['MENTHLTH'] >= 14
    poor_ovr_cond = recoded_data['POORHLTH'] >= 14
    persdoc_cond = (recoded_data['PERSDOC'] == 1) | \
        (recoded_data['PERSDOC'] == 2)
    checkup_cond = recoded_data['CHECKUP'] == 1
    # Create new binary variables defined as described in function docstring.
    recoded_data['GOOD_GENHLTH'] = np.where(genhlth_cond, 0, 1).astype('int64')
    recoded_data['PHYS_DISTRESS'] = \
        np.where(phys_distress_cond, 1, 0).astype('int64')
    recoded_data['MENT_DISTRESS'] = \
        np.where(ment_distress_cond, 1, 0).astype('int64')
    recoded_data['POOR_OVR_HLTH'] = \
        np.where(poor_ovr_cond, 1, 0).astype('int64')
    recoded_data['HAS_PERSDOC'] = \
        np.where(persdoc_cond, 1, 0).astype('int64')
    recoded_data['ANNUAL_CHECKUP'] = \
        np.where(checkup_cond, 1, 0).astype('int64')
    return recoded_data


def calculate_proportion(data, groupings, outcome):
    """
    Given a pandas data frame as data, a list of string column names in
    groupings, and a another column that represents a characteristic in
    outcome, groups the data according to the given groupings and then
    calculates the proportion of each each group that displays the given
    outcome.

    Arguments:
        data: A pandas data frame
        groupings: a list of column names from data from which to group by.
        outcome: the outcome being represented which is the proportion of
                 interest.

    Returns:
        A pandas series of length of the original data frame, designed to be
        stored as a column back into the original data frame.
    """
    # Derive number of observations that exhibit outcome by groupings.
    count = data.groupby(groupings)[outcome].transform('sum')
    # Derive total number of observations per outcome by groupings.
    total_counts = data.groupby(groupings)[outcome].transform('count')
    proportion = count / total_counts
    return proportion


def get_proportions_data(total_data):
    """
    Given BRFSS data in a pandas data frame, recoded and with proportions prep
    done, calculates and returns a pandas data frame that has all original
    input columns but also additional variables, one for each outcome, in
    which the proportion of each binary outcome being 1 (i.e., the outcome
    is observed) is reported relative to total sample size according to
    treatment status and the year. Thus, we get a proportional representation
    across all years of the number of individuals in both treatment and control
    who experience the outcome.

    Arguments:
        total_data: A pandas data frame of the format returned by
                    proportions_prep_for_outcomes.

    Returns:
        A pandas data frame with proportional variables by year and status of
        TREAT.
    """
    # Identify outcomes
    outcomes = ['GOOD_GENHLTH', 'PHYS_DISTRESS', 'MENT_DISTRESS',
                'POOR_OVR_HLTH', 'HLTHPLN', 'HAS_PERSDOC', 'MEDCOST',
                'ANNUAL_CHECKUP']
    copied = total_data.copy()
    for outcome in outcomes:
        # Compute proportions of individuals who experienced the outcome
        # for each year in each treatment group.
        proportion = calculate_proportion(copied,
                                          ['YEAR', 'TREAT_EARLY',
                                           'TREAT_LATE'],
                                          outcome)
        copied['PROP_' + outcome] = proportion
        # Now, for the sake of graphing parallel trends, create 'TREAT'
        # indicator, and compute proportions based solely on treatment.
        treat_cond = (copied['TREAT_EARLY'] == 1) | \
            (copied['TREAT_LATE'] == 1)
        copied['TREAT'] = np.where(treat_cond, 1, 0)
        proportion = calculate_proportion(copied, ['YEAR', 'TREAT'],
                                          outcome)
        copied['ALL_PROP_' + outcome] = proportion
    return copied


def parallel_trends(prop_data):
    """
    Given the prop_data returned by get_proportions_data, checks for
    parallel trends across all outcomes, pre-2018.

    Arguments:
        prop_data: A pandas data frame of the format returned by the
                   get_proportions_data method.
        states: A string argument representing which states to look at
                trends for. "all" indicates using all data available,
                "2018" indicates using treatment states that implemented
                reinsurance in 2018 and "2019" does the same for 2019 data.

    Returns:
        None.
    """
    current_wd = os.getcwd()
    # Create directory to store graphs, should it not exist yet.
    if not os.path.exists(current_wd + '/parallel_trends_graphs'):
        os.mkdir(current_wd + '/parallel_trends_graphs')
    # Define outcomes and graph titles.
    outcomes = ['ALL_PROP_GOOD_GENHLTH', 'ALL_PROP_PHYS_DISTRESS',
                'ALL_PROP_MENT_DISTRESS', 'ALL_PROP_POOR_OVR_HLTH',
                'ALL_PROP_HLTHPLN', 'ALL_PROP_HAS_PERSDOC',
                'ALL_PROP_MEDCOST', 'ALL_PROP_ANNUAL_CHECKUP']
    titles = ['Reported good general health', 'Reported physical distress',
              'Reported mental distress', 'Reported overall poor health',
              'Had a health plan', 'Had a personal doctor',
              'Could not see a doctor due to cost',
              'Had a checkup within the past year']
    # Store each outcome/outcome graph title as a tuple in a list.
    graph_list = []
    for i in range(len(outcomes)):
        entry = (outcomes[i], titles[i])
        graph_list.append(entry)
    for j in range(len(graph_list)):
        outcome = graph_list[j][0]
        graph_title = 'Proportion of individuals who ' + graph_list[j][1]
        graph_data = prop_data.copy()
        # Grab only columns of interest
        graph_data = graph_data[[outcome, 'YEAR', 'TREAT']]
        # Drop duplicates to prevent plotly complications.
        graph_data = graph_data.drop_duplicates(subset=[outcome])
        # Omit 2020 data (too few observations, which are contained within
        # the 2019 BRFSS survey file).
        latest_year = graph_data['YEAR'] == 2020
        graph_data = graph_data[~latest_year]
        # Only graph pre-treatment trends.
        year_mask = graph_data['YEAR'] <= 2017
        graph_data = graph_data[year_mask]
        # Convert year variable to a date-time format for graphical clarity.
        graph_data['YEAR'] = pd.to_datetime(graph_data['YEAR'],
                                            format='%Y')
        fig = px.line(graph_data, x='YEAR', y=outcome,
                      color='TREAT',
                      labels=dict(color='Treatment'), title=graph_title)
        fig.update_xaxes(title_text='Year')
        # Determine y-axis limits, dependent upon range of outcome values.
        upper_limit = graph_data[outcome].max()
        # Specify that all proportions must exist between a [0,1] interval for
        # good bounds on graph axes.
        upper_axe_bound = round(upper_limit + 0.25, 1)
        if upper_axe_bound > 1.0:
            upper_axe_bound = 1.0
        lower_limit = graph_data[outcome].min()
        lower_axe_bound = round(lower_limit - 0.25, 1)
        if lower_axe_bound < 0.0:
            lower_axe_bound = 0.0
        axe_limits = [lower_axe_bound, upper_axe_bound]
        fig.update_yaxes(range=axe_limits, title_text='proportion (%) of'
                                                      'individuals')
        fig.show()
        fig.write_image('parallel_trends_graphs/' + outcome + '.png')


def main():
    current_wd = os.getcwd()
    rewrite_data(current_wd)
    all_data = clean_data(current_wd)
    state_data = []
    state_ids = [2, 23, 24, 41, 27, 55, 42, 53, 26, 36, 4, 56, 46, 20]
    for state in state_ids:
        state_datasets = acquire_state_data(all_data, state)
        reweighted_states = weight_state_data(state_datasets)
        all_state_data = consolidate_data(reweighted_states)
        state_data.append(all_state_data)
    complete_data = consolidate_data(state_data)
    multi_time_data = construct_additional_vars(complete_data)
    multi_time_data.to_csv(current_wd + '/multi_time_data.csv')
    recoded = binary_recode(multi_time_data)
    recoded = proportions_prep_for_outcomes(recoded)
    all_prop_data = get_proportions_data(recoded)
    all_prop_data.to_csv(current_wd + '/multi_time_prop_data.csv')
    parallel_trends(all_prop_data)


if __name__ == '__main__':
    main()
