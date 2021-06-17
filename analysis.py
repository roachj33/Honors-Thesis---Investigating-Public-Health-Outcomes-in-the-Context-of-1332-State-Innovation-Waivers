import pandas as pd
import numpy as np
import os
import plotly.express as px
from patsy import dmatrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, \
    LogisticRegressionCV
from sklearn.metrics import mean_squared_error, r2_score


def recode_variables(total_data):
    """
    Given the complete_data.csv file in a pandas data frame, recodes some
    variables in the following ways: the outcome variables PHYSHLTH, MENTHLTH,
    POORHLTH, and CHILDREN have their values '88' recoded to 0 since an 88
    means "none", and we can otherwise treat these variables as quantitative.
    For the variables HLTHPLN, MEDCOST, HRTDIS, HRTATTCK, STROKE, SKNCNCR,
    OTHERCNCR, KIDDIS, COPD, ARTH, and DEPRESS converts '2.0' to '0.0', where
    a 2.0 means "no" in the survey, and we are only switching to the more
    traditional binary format. The variable BMI currently has a '1' for no
    and a '2' for yes, so switch '2' to a '1' and '1' to a '0'. Also, converts
    these and any other variables in float format to integer format for
    simplifying purposes.

    Arguments:
        total_data: A pandas data frame of the complete_data.csv.

    Returns:
        An updated pandas data frame with the alterations mentioned above.
    """
    # Drop pointless index column
    total_data = total_data.drop('Unnamed: 0', axis=1)
    # Variables to convert 88's to 0's.
    variables_88 = ['PHYSHLTH', 'MENTHLTH', 'POORHLTH', 'CHILDREN']
    for variable in variables_88:
        total_data[variable] = \
             total_data[variable].replace(88, 0).astype('int64')
    # Variables to convert 2's to 0's to match up with standard binary format.
    # Also, changes these variables to integer format.
    variables_binary = ['HLTHPLN', 'MEDCOST', 'HRTDIS', 'HRTATTCK', 'STROKE',
                        'SKNCNCR', 'OTHERCNCR', 'KIDDIS', 'COPD', 'ARTH',
                        'DEPRESS']
    for variable in variables_binary:
        total_data[variable] = \
            total_data[variable].replace(2, 0).astype('int64')
    # BMI is weird so deal with it independently.
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
    recode_variables, simplifies outcome variables down to binary
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
    # Get conditions for GOOD_GENHLTH variable. Use false conditions.
    genhlth_cond = (recoded_data['GENHLTH'] == 4) | \
        (recoded_data['GENHLTH'] == 5)
    # Get conditions for PHYS_DISTRESS, MENT_DISTRESS, and POOR_OVR_HLTH.
    phys_distress_cond = recoded_data['PHYSHLTH'] >= 14
    ment_distress_cond = recoded_data['MENTHLTH'] >= 14
    poor_ovr_cond = recoded_data['POORHLTH'] >= 14
    # Get conditions for HAS_PERSDOC.
    persdoc_cond = (recoded_data['PERSDOC'] == 1) | \
        (recoded_data['PERSDOC'] == 2)
    # Get conditions for ANNUAL_CHECKUP.
    checkup_cond = recoded_data['CHECKUP'] == 1
    # Create new binary variables as described in function description.
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


def get_proportions_data(total_data, states, multi=False):
    """
    Given BRFSS data in a pandas data frame, recoded and with proportions prep
    done, calculates and returns a pandas data frame that has all original
    input columns but also additional variables, one for each outcome, in
    which the proportion of each binary outcome being 1 (i.e., the outcome
    is observed) is reported relative to total sample size according to
    treatment status and the year. Thus, we get a proportional representation
    across all years of the number of individuals in both treatment and control
    who experience the outcome. States is a string variables taking possible
    values of 'all', '2018', '2019', or  'wisc', indicating calculation of
    proportions for all data, only 2018 treatment and control groups, only
    2019 treatment and control groups, or omission of Wisconsin and selected
    non-medicaid expansion control states.

    Arguments:
        total_data: A pandas data frame of the format returned by
                    proportions_prep_for_outcomes.
        states: A string variable taking possible values of 'all', '2018',
                '2019' or 'wisc', each slightly changing which data we are
                operating on as defined above.

    Returns:
        A pandas data frame with proportional variables by year and status of
        TREAT.
    """
    # Sub-population masks.
    # low_income = (total_data['INCOME'] == 1) | (total_data['INCOME'] == 2) |\
    #              (total_data['INCOME'] == 3) | (total_data['INCOME'] == 4)
    # hrt_dis = total_data['CHR_DIS'] == 1
    # depress = total_data['DEPRESS'] == 1
    # cancer = (total_data['SKNCNCR'] == 1) | (total_data['OTHERCNCR'] == 1)
    # Identify outcomes
    outcomes = ['GOOD_GENHLTH', 'PHYS_DISTRESS', 'MENT_DISTRESS',
                'POOR_OVR_HLTH', 'HLTHPLN', 'HAS_PERSDOC', 'MEDCOST',
                'ANNUAL_CHECKUP']
    copied = total_data.copy()
    # copied = copied[cancer]
    if states == '2018':
        # Omit 2019 treatment states and non-medicaid expansion controls.
        id_mask = (total_data['STATE'] == 23) | (total_data['STATE'] == 24) | \
                  (total_data['STATE'] == 55) | (total_data['STATE'] == 47) | \
                  (total_data['STATE'] == 46) | (total_data['STATE'] == 20)
        copied = copied[~id_mask]
    elif states == '2019':
        id_mask = (total_data['STATE'] == 41) | (total_data['STATE'] == 2) | \
                  (total_data['STATE'] == 27)
        copied = copied[~id_mask]
    elif states == 'wisc':
        id_mask = (total_data['STATE'] == 55) | (total_data['STATE'] == 47) | \
                  (total_data['STATE'] == 46) | (total_data['STATE'] == 20)
        copied = copied[~id_mask]
    else:
        pass
    if multi is False:
        for outcome in outcomes:
            proportion = calculate_proportion(copied, ['YEAR', 'TREAT'],
                                              outcome)
            copied['PROP_' + outcome] = proportion
    else:
        for outcome in outcomes:
            proportion = calculate_proportion(copied,
                                              ['YEAR', 'TREAT_EARLY',
                                               'TREAT_LATE'],
                                              outcome)
            copied['PROP_' + outcome] = proportion
            # Now, for the sake of graphing parallel trends, create 'TREAT'
            # indicator.
            treat_cond = (copied['TREAT_EARLY'] == 1) | \
                (copied['TREAT_LATE'] == 1)
            copied['TREAT'] = np.where(treat_cond, 1, 0)
            proportion = calculate_proportion(copied, ['YEAR', 'TREAT'],
                                              outcome)
            copied['ALL_PROP_' + outcome] = proportion
    print(len(copied))
    return copied


def parallel_trends(prop_data, multi=False):
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
    if multi is False:
        outcomes = ['PROP_GOOD_GENHLTH', 'PROP_PHYS_DISTRESS',
                    'PROP_MENT_DISTRESS', 'PROP_POOR_OVR_HLTH', 'PROP_HLTHPLN',
                    'PROP_HAS_PERSDOC', 'PROP_MEDCOST', 'PROP_ANNUAL_CHECKUP']
    else:
        outcomes = ['ALL_PROP_GOOD_GENHLTH', 'ALL_PROP_PHYS_DISTRESS',
                    'ALL_PROP_MENT_DISTRESS', 'ALL_PROP_POOR_OVR_HLTH',
                    'ALL_PROP_HLTHPLN', 'ALL_PROP_HAS_PERSDOC',
                    'ALL_PROP_MEDCOST', 'ALL_PROP_ANNUAL_CHECKUP']
    titles = ['reported good general health', 'reported physical distress',
              'reported mental distress', 'reported overall poor health',
              'had a health plan', 'had a personal doctor',
              'needed to see a medical doctor, but could not due to cost',
              'had a checkup within the past year']
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
        # Omit 2020 data (too few observations)
        latest_year = graph_data['YEAR'] == 2020
        graph_data = graph_data[~latest_year]
        # Only graph pre-treatment trends.
        year_mask = graph_data['YEAR'] <= 2017
        graph_data = graph_data[year_mask]
        # Convert year variable to a date-time format.
        graph_data['YEAR'] = pd.to_datetime(graph_data['YEAR'],
                                            format='%Y')
        # Create graph.
        fig = px.line(graph_data, x='YEAR', y=outcome,
                      color='TREAT',
                      labels=dict(color='Treatment'), title=graph_title)
        fig.update_xaxes(title_text='Year')
        # Determine y-axis limits, dependent upon range of outcome values.
        upper_limit = graph_data[outcome].max()
        # Specify that all proportions must exist between [0,1] interval.
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
        # Show graph, save it in previously created directory.
        fig.show()
        fig.write_image('parallel_trends_graphs/' + outcome + '.png')


def get_basic_model_components(recoded_data, binary=False):
    """
    Given a data frame of recoded BRFSS data of the form returned by
    recode_variables, grabs the features and outcomes for the most basic
    research model developed, creating dummies for the features if necessary
    and keeping each outcome stored as a series in a list. Returns a two-tuple
    where the first index is the pandas data frame of features, and the second
    is a list where each entry in the list is an outcome in the form of a
    pandas series.

    Arguments:
        recoded_data: The pandas data frame returned by recode_variables().

    Returns:
        A 2-tuple where the first entry is a pandas data frame with the model
        features and the second entry is a list, whose entries represent model
        outcomes in the form of pandas series.
    """
    year_mask = recoded_data['YEAR'] == 2020
    recoded_data = recoded_data[~year_mask]
    # Outcome variables of interest
    if binary is False:
        outcomes = ['PROP_GOOD_GENHLTH', 'PROP_PHYS_DISTRESS',
                    'PROP_MENT_DISTRESS', 'PROP_POOR_OVR_HLTH', 'PROP_HLTHPLN',
                    'PROP_HAS_PERSDOC', 'PROP_MEDCOST', 'PROP_ANNUAL_CHECKUP']
        # Additional variables to drop for most basic research model.
        additional_drops = ['HRTDIS', 'HRTATTCK', 'STROKE', 'SKNCNCR',
                            'OTHERCNCR', 'BMI', 'WEIGHT', 'KIDDIS', 'COPD',
                            'ARTH', 'DEPRESS', 'DIABETE', 'GENHLTH',
                            'PHYSHLTH', 'MENTHLTH', 'POORHLTH', 'HLTHPLN',
                            'PERSDOC', 'MEDCOST', 'CHECKUP', 'GOOD_GENHLTH',
                            'PHYS_DISTRESS', 'MENT_DISTRESS', 'POOR_OVR_HLTH',
                            'HAS_PERSDOC', 'ANNUAL_CHECKUP', 'EDUCA']
    else:
        outcomes = ['GOOD_GENHLTH', 'PHYS_DISTRESS', 'MENT_DISTRESS',
                    'POOR_OVR_HLTH', 'HLTHPLN', 'HAS_PERSDOC', 'MEDCOST',
                    'ANNUAL_CHECKUP']
        additional_drops = ['HRTDIS', 'HRTATTCK', 'STROKE', 'SKNCNCR',
                            'OTHERCNCR', 'BMI', 'WEIGHT', 'KIDDIS', 'COPD',
                            'ARTH', 'DEPRESS', 'DIABETE', 'GENHLTH',
                            'PHYSHLTH', 'MENTHLTH', 'POORHLTH', 'HLTHPLN',
                            'PERSDOC', 'PROP_MEDCOST', 'CHECKUP',
                            'PROP_GOOD_GENHLTH', 'PROP_PHYS_DISTRESS',
                            'PROP_MENT_DISTRESS', 'PROP_POOR_OVR_HLTH',
                            'PROP_HAS_PERSDOC', 'PROP_ANNUAL_CHECKUP', 'EDUCA']
    # Drop outcomes and additional variables from features data frame.
    to_drop = outcomes + additional_drops
    features = recoded_data.drop(to_drop, axis=1)
    # Variables that need to be encoded as dummy variables.
    dummies = ['STATE', 'MARITAL', 'INCOME', 'RENT', 'AGE', 'SEX',
               'EMPLOY', 'RACE', 'TREAT', 'TIME', 'CHR_DIS', 'YEAR']
    # Get needed dummy features.
    features = pd.get_dummies(features, columns=dummies)
    # Time to get outcomes as pandas series.
    # Initialize list to store each series.
    outcomes_list = []
    for outcome in outcomes:
        data = recoded_data[outcome]
        outcomes_list.append(data)
    # Store features and list of outcomes in tuple.
    result = (features, outcomes_list)
    return result


def obtain_model_splits(feats_and_labels):
    """
    Given a two-tuple where the first entry is features for the research
    model and the second is the outcomes for the model, as in the kind
    returned by get_basic_model_components, obtains a training/testing
    split for modeling purposes.

    Arguments:
        feats_and_labels: A two-tuple of the kind returned by
        get_basic_model_components

    Returns:
        A list of length 8, each list entry corresponding to a four-tuple,
        with entry one of the tuple being training features, two being
        training outcomes, three being testing features and four being
        testing outcomes.
    """
    # Determine the number of outcomes.
    num_outcomes = len(feats_and_labels[1])
    data_list = []
    feats = feats_and_labels[0]
    for i in range(num_outcomes):
        label = feats_and_labels[1][i]
        feat_train, feat_test, label_train, label_test = \
            train_test_split(feats, label, train_size=0.8)
        components = (feat_train, feat_test, label_train, label_test)
        data_list.append(components)
    return data_list


def get_basic_data_stats(all_data):
    """
    Given a pandas data frame, containing annual BRFSS data, calculates
    the sample size, in total and for each year.

    Arguments:
        all_data: A pandas dataframe containing BRFSS data.

    Returns:
        None.
    """
    sample_size = len(all_data)
    print('There are', sample_size, 'observations in total')
    years = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    ids = all_data['STATE'].unique()
    for year in years:
        year_mask = all_data['YEAR'] == year
        annual_data = all_data[year_mask]
        annual_sample = len(annual_data)
        print('There are', annual_sample, 'observations in year', year)
        for state_id in ids:
            state_mask = annual_data['STATE'] == state_id
            state_data = annual_data[state_mask]
            state_sample = len(state_data)
            print('There are', state_sample, 'observations in year', year,
                  'for state', state_id)


def acquire_reg_model(features):
    """
    Given a pandas data frame containing features of the diff-in-diff model,
    creates and returns a DesignMatrix representing the formula for the
    regressor.

    Arguments:
        features: A pandas data frame of BRFSS features.

    Returns:
        A DesignMatrix object which is the formula to regress outcomes on.
    """
    # Indicate formula to use for regression, starting with state and then
    # year fixed effects, then sociodemographic controls, then treatment
    # and time indicators and then the interaction between treatment and
    # time.
    formula = '(STATE_4 + STATE_20 + STATE_23 + STATE_24 +' \
        ' STATE_26 + STATE_27 + STATE_36 + STATE_41 + STATE_42 + STATE_46 +' \
        ' STATE_47 + STATE_53 + STATE_55) + (YEAR_2013 +' \
        ' YEAR_2014 + YEAR_2015 + YEAR_2016 + YEAR_2017 + YEAR_2018 +' \
        ' YEAR_2019) + (MARITAL_2 + MARITAL_3 + MARITAL_4 +' \
        ' MARITAL_5 + MARITAL_6 + INCOME_2 + INCOME_3 + INCOME_4' \
        ' + INCOME_5 + INCOME_6 + INCOME_7 + INCOME_8 + RENT_2 +' \
        ' RENT_3 + AGE_1 + SEX_2 + EMPLOY_2 + EMPLOY_3 + EMPLOY_4 +' \
        ' EMPLOY_5 + EMPLOY_6 + EMPLOY_7 + EMPLOY_8 +' \
        ' RACE_2 + RACE_3 + RACE_4 + RACE_5) + (TREAT_1 * TIME_1)'
    matrix = dmatrix(formula, features, return_type='dataframe')
    return matrix


def run_ols_reg(ml_data, weight=False):
    """
    Given a list containing tuples of BRFSS data prepped for machine learning,
    like of the form returned by obtain_model_splits(), conducts an OLS
    regression of each outcome on the diff-in-diff model. Returns a list
    of tuples, each tuple corresponding to a different regression and
    containing coefficient estimates and testing MSE in each entry.

    Arguments:
        ml_data: A list of tuples of BRFSS data prepped for machine learning,
                 as returned by obtain_model_splits.
        weight: A boolean variable indicating whether to utilize weights
                in analysis or not. Default behavior is false.

    Returns:
        A list of tuples containing coefficient estimates and testing MSE for
        each model.
    """
    results = []
    for data in ml_data:
        feat_train = data[0]
        feat_test = data[1]
        label_train = data[2]
        label_test = data[3]
        feat_train_matrix = acquire_reg_model(feat_train)
        feat_test_matrix = acquire_reg_model(feat_test)
        reg = LinearRegression()
        if weight is True:
            train_weights = feat_train['REWEIGHT']
            test_weights = feat_test['REWEIGHT']
            reg.fit(feat_train_matrix, label_train,
                    sample_weight=train_weights)
            predictions = reg.predict(feat_test_matrix)
            mse = mean_squared_error(label_test, predictions,
                                     sample_weight=test_weights)
            coef_of_det = r2_score(predictions, label_test,
                                   sample_weight=test_weights)
        else:
            reg.fit(feat_train_matrix, label_train)
            predictions = reg.predict(feat_test_matrix)
            mse = mean_squared_error(label_test, predictions)
            coef_of_det = r2_score(predictions, label_test)
        model_coefs = pd.DataFrame(reg.coef_, feat_train_matrix.columns,
                                   columns=['Coefficients'])
        model_coefs = model_coefs.apply(lambda x: '%10f' % x, axis=1)
        print(predictions)
        # coefs = model_coefs['Coefficients']
        # model_coefs['Coefficients'] = coefs.round(7)
        # model_coefs['Coefficients'] = f'{coefs:.5f}'
        model_outcomes = (model_coefs, mse, coef_of_det)
        results.append(model_outcomes)
    return results


def run_lasso_reg(ml_data, weight=False):
    """
    Given a list containing tuples of BRFSS data prepped for machine learning,
    like of the form returned by obtain_model_splits(), conducts an OLS
    regression of each outcome on the diff-in-diff model. Returns a list
    of tuples, each tuple corresponding to a different regression and
    containing coefficient estimates and testing MSE in each entry.

    Arguments:
        ml_data: A list of tuples of BRFSS data prepped for machine learning,
                 as returned by obtain_model_splits.
        weight: A boolean variable indicating whether to utilize weights
                in analysis or not. Default behavior is false.

    Returns:
        A list of tuples containing coefficient estimates and testing MSE for
        each model.
    """
    results = []
    for data in ml_data:
        feat_train = data[0]
        feat_test = data[1]
        label_train = data[2]
        label_test = data[3]
        feat_train_matrix = acquire_reg_model(feat_train)
        feat_test_matrix = acquire_reg_model(feat_test)
        reg = LassoCV(cv=5)
        if weight is True:
            train_weights = feat_train['REWEIGHT']
            test_weights = feat_test['REWEIGHT']
            reg.fit(feat_train_matrix, label_train,
                    sample_weight=train_weights)
            predictions = reg.predict(feat_test_matrix)
            mse = mean_squared_error(label_test, predictions,
                                     sample_weight=test_weights)
            coef_of_det = r2_score(predictions, label_test,
                                   sample_weight=test_weights)
        else:
            reg.fit(feat_train_matrix, label_train)
            predictions = reg.predict(feat_test_matrix)
            mse = mean_squared_error(label_test, predictions)
            coef_of_det = r2_score(predictions, label_test)
        model_coefs = pd.DataFrame(reg.coef_, feat_train_matrix.columns,
                                   columns=['Coefficients'])
        model_coefs = model_coefs.apply(lambda x: '%10f' % x, axis=1)
        print(predictions)
        # coefs = model_coefs['Coefficients']
        # model_coefs['Coefficients'] = coefs.round(7)
        # model_coefs['Coefficients'] = f'{coefs:.5f}'
        model_outcomes = (model_coefs, mse, coef_of_det)
        results.append(model_outcomes)
    return results


def run_logistic_reg(ml_data, weight=False):
    """
    Given a list containing tuples of BRFSS data prepped for machine learning,
    like of the form returned by obtain_model_splits(), conducts an OLS
    regression of each outcome on the diff-in-diff model. Returns a list
    of tuples, each tuple corresponding to a different regression and
    containing coefficient estimates and testing MSE in each entry.

    Arguments:
        ml_data: A list of tuples of BRFSS data prepped for machine learning,
                 as returned by obtain_model_splits.
        weight: A boolean variable indicating whether to utilize weights
                in analysis or not. Default behavior is false.

    Returns:
        A list of tuples containing coefficient estimates and testing MSE for
        each model.
    """
    results = []
    for data in ml_data:
        feat_train = data[0]
        feat_test = data[1]
        label_train = data[2]
        label_test = data[3]
        feat_train_matrix = acquire_reg_model(feat_train)
        feat_test_matrix = acquire_reg_model(feat_test)
        reg = LogisticRegressionCV(cv=5)
        if weight is True:
            train_weights = feat_train['REWEIGHT']
            test_weights = feat_test['REWEIGHT']
            reg.fit(feat_train_matrix, label_train,
                    sample_weight=train_weights)
            predictions = reg.predict(feat_test_matrix)
            mse = mean_squared_error(label_test, predictions,
                                     sample_weight=test_weights)
            coef_of_det = r2_score(predictions, label_test,
                                   sample_weight=test_weights)
        else:
            reg.fit(feat_train_matrix, label_train)
            predictions = reg.predict(feat_test_matrix)
            mse = mean_squared_error(label_test, predictions)
            coef_of_det = r2_score(predictions, label_test)
        print(len(reg.coef_))
        print(len(feat_test_matrix.columns))
        model_coefs = pd.DataFrame(np.transpose(reg.coef_),
                                   feat_train_matrix.columns,
                                   columns=['Coefficients'])
        model_coefs = model_coefs.apply(lambda x: '%10f' % x, axis=1)
        print(predictions)
        # coefs = model_coefs['Coefficients']
        # model_coefs['Coefficients'] = coefs.round(7)
        # model_coefs['Coefficients'] = f'{coefs:.5f}'
        model_outcomes = (model_coefs, mse, coef_of_det)
        results.append(model_outcomes)
    return results


def main():
    """
    Conducts my research analysis.
    """
    print('Starting analysis')
    # data = pd.read_csv('complete_data.csv')
    data = pd.read_csv('multi_time_data.csv')
    print(list(data.columns))
    # data = pd.read_csv('complete_data.csv')
    # print(len(data))
    # print(list(data.columns))
    # print(sorted(data.DIABETE.unique()))
    recoded = recode_variables(data)
    recoded = proportions_prep_for_outcomes(recoded)
    # print(list(recoded.columns))
    # get_basic_model_components(recoded)
    # prop_2018_data = get_proportions_data(recoded, '2018')
    # prop_2019_data = get_proportions_data(recoded,y '2019')
    # prop_no_wisc_data = get_proportions_data(recoded, 'wisc')
    # all_prop_data = get_proportions_data(recoded, 'all')
    all_prop_data = get_proportions_data(recoded, 'all', multi=True)
    current_wd = os.getcwd()
    # all_prop_data.to_csv(current_wd + '/all_prop_data.csv')
    all_prop_data.to_csv(current_wd + '/multi_time_prop_data.csv')
    # parallel_trends(prop_2018_data)
    # parallel_trends(prop_2019_data)
    # parallel_trends(prop_no_wisc_data)
    parallel_trends(all_prop_data, multi=True)
    # print(list(prop_data.columns))
    # get_basic_data_stats(all_prop_data)
    # basic_data = get_basic_model_components(all_prop_data)
    # basic_binary_data = get_basic_model_components(all_prop_data, binary=True
    # print(len(basic_data[0]))
    # print(basic_data[0])
    # print(len(basic_data[1]))
    # for entry in basic_data[1]:
    #     print(entry)
    # ml_data = obtain_model_splits(basic_data)
    # ml_binary_data = obtain_model_splits(basic_binary_data)
    # results = run_ols_reg(ml_data)
    # results = run_lasso_reg(ml_data)
    # binary_results = run_logistic_reg(ml_binary_data)
    # for result in results:
    # coefs = result[0]
    # mse = result[1]
    # coef_of_det = result[2]
    # print(coefs)
    # print(coefs.dtypes)
    # print(mse)
    # print(coef_of_det)
    # ex = ml_data[0][0]
    # print(list(ex.columns))
    # acquire_reg_model('blah')
    print('Analysis is complete')


if __name__ == '__main__':
    main()
