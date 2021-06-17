"""
Name: Joey Roach
Date: May 17th, 2021
test.py is a python script which tests the methods found in simplify.py.
"""


from simplify import recode_data, check_rewrite, weight_state_data, \
                get_state_obs, construct_additional_vars, binary_recode, \
                proportions_prep_for_outcomes, get_proportions_data
import pandas as pd
import numpy as np
import os
from pandas.testing import assert_frame_equal


def test_rewrite_data(current_wd):
    """
    Tests that the rewrite_data function returned consistent column names
    for all written csv files, and that each file contains the expected number
    of observations.
    """
    expected_names = {'_STATE', 'IYEAR', 'NUMADULT', 'GENHLTH', 'PHYSHLTH',
                      'MENTHLTH', 'POORHLTH', 'HLTHPLN1', 'PERSDOC2',
                      'MEDCOST', 'CHECKUP1', 'CVDCRHD4', 'CVDINFR4',
                      'CVDSTRK3', 'CHCSCNCR', 'CHCOCNCR', 'CHCCOPD1',
                      'HAVARTH3', 'ADDEPEV2', 'CHCKIDNY', 'DIABETE3',
                      'SEX', 'MARITAL', 'EDUCA', 'RENTHOM1', 'EMPLOY1',
                      'CHILDREN', 'INCOME2', '_RACE_G1', '_RFBMI5',
                      '_AGE65YR', 'Unnamed: 0', '_LLCPWT', 'ASTHNOW',
                      'FLSHTMY2', 'HIVTST6', 'HIVTSTD3', '_STSTR'}
    rewritten = check_rewrite(current_wd)
    expected_obs = [475687, 491773, 464664, 441456, 486303, 450016,
                    437436, 418268]
    for i in range(len(expected_obs)):
        col_names = rewritten[i][0]
        data_size = rewritten[i][1]
        assert (col_names == expected_names)
        assert(data_size == expected_obs[i])
    print('passed rewrite tests!')


def test_recode():
    """
    Tests the recode_data method on user-generated data.
    """
    test_one = {'STATE': [1, 1, 1], 'YEAR': [2012, 2012, 2012],
                'NUMADULT': [4, 9, 7], 'GENHLTH': [7, 9, 1],
                'PHYSHLTH': [1, 77, 99], 'MENTHLTH': [99, 2, 77],
                'POORHLTH': [77, 99, 77], 'HLTHPLN': [7, 9, 2],
                'PERSDOC': [7, 3, 9], 'MEDCOST': [7, 7, 9],
                'CHECKUP': [7, 1, 9], 'HRTDIS': [9, 9, 7],
                'HRTATTCK': [1, 9, 7], 'STROKE': [9, 9, 7],
                'SKNCNCR': [3, 7, 9], 'OTHERCNCR': [1, 9, 7],
                'MARITAL': [1, 7, 9], 'EDUCA': [4, 9, 7],
                'RENT': [7, 7, 9], 'CHILDREN': [99, 1, 77],
                'INCOME': [99, 77, 1], 'AGE': [3, 5, 7],
                'BMI': [9, 9, 1], 'WEIGHT': [9, 7, 99],
                'COPD': [7, 9, 1], 'ARTH': [9, 7, 2],
                'KIDDIS': [9, 7, 9], 'DIABETE': [7, 7, 9],
                'SEX': [9, 1, 7], 'EMPLOY': [9, 7, 1],
                'RACE': [1, 7, 9], 'DEPRESS': [9, 7, 1]}
    expected_one = {'STATE': [1, 1, 1], 'YEAR': [2012, 2012, 2012],
                    'NUMADULT': [4, 9, 7], 'GENHLTH': [np.nan, np.nan, 1],
                    'PHYSHLTH': [1, np.nan, np.nan],
                    'MENTHLTH': [np.nan, 2, np.nan],
                    'POORHLTH': [np.nan, np.nan, np.nan],
                    'HLTHPLN': [np.nan, np.nan, 2],
                    'PERSDOC': [np.nan, 3, np.nan],
                    'MEDCOST': [np.nan, np.nan, np.nan],
                    'CHECKUP': [np.nan, 1, np.nan],
                    'HRTDIS': [np.nan, np.nan, np.nan],
                    'HRTATTCK': [1, np.nan, np.nan],
                    'STROKE': [np.nan, np.nan, np.nan],
                    'SKNCNCR': [3, np.nan, np.nan],
                    'OTHERCNCR': [1, np.nan, np.nan],
                    'MARITAL': [1, np.nan, np.nan],
                    'EDUCA': [4, np.nan, np.nan],
                    'RENT': [np.nan, np.nan, np.nan],
                    'CHILDREN': [np.nan, 1, 77],
                    'INCOME': [np.nan, np.nan, 1], 'AGE': [np.nan, 5, 7],
                    'BMI': [np.nan, np.nan, 1], 'WEIGHT': [9, 7, 99],
                    'COPD': [np.nan, np.nan, 1], 'ARTH': [np.nan, np.nan, 2],
                    'KIDDIS': [np.nan, np.nan, np.nan],
                    'DIABETE': [np.nan, np.nan, np.nan],
                    'SEX': [np.nan, 1, np.nan], 'EMPLOY': [np.nan, 7, 1],
                    'RACE': [1, 7, 9], 'DEPRESS': [np.nan, np.nan, 1]}
    expected_result = pd.DataFrame(data=expected_one)
    test_df = pd.DataFrame(data=test_one)
    actual_result = recode_data(test_df)
    assert_frame_equal(actual_result, expected_result)
    print('passed recode tests!')


def test_get_state_obs():
    """
    Tests the get_state_obs method.
    """
    test_data_one = {'obs': [1, 5, 8, 9]}
    test_data_two = {'obs': [7, 9, 8, 3, 5, 7, 9]}
    test_data_three = {'obs': [0, 1, 2, 3, 4, 5]}
    df_one = pd.DataFrame(data=test_data_one)
    df_two = pd.DataFrame(data=test_data_two)
    df_three = pd.DataFrame(data=test_data_three)
    test_list = [df_one, df_two, df_three]
    expected_result = 17
    actual_result = get_state_obs(test_list)
    assert(actual_result == expected_result)
    print('passed total state observation tests!')


def test_weight_state_data():
    """
    Tests the weight_state_data method on user-generated input, ensuring
    the function output does follow BRFSS re-weighting procedures.
    """
    test_2012 = {'YEAR': [2012, 2012, 2012],
                 'WEIGHT': [0.5, 0.2, 0.4]}
    test_2013 = {'YEAR': [2013],
                 'WEIGHT': [1.5]}
    test_2014 = {'YEAR': [2014, 2014],
                 'WEIGHT': [1.6, 0.8]}
    test_2015 = {'YEAR': [2015, 2015, 2015, 2015, 2015],
                 'WEIGHT': [3.4, 4.8, 5.0, 6.1, 3.4]}
    test_2016 = {'YEAR': [2016, 2016],
                 'WEIGHT': [4.1, 5.3]}
    test_2017 = {'YEAR': [2017],
                 'WEIGHT': [0.4]}
    test_2018 = {'YEAR': [2018, 2018, 2018, 2018],
                 'WEIGHT': [0.9, 7.5, 5.5, 1.0]}
    test_2019 = {'YEAR': [2019, 2019],
                 'WEIGHT': [2.2, 0.5]}
    expected_2012 = {'YEAR': [2012, 2012, 2012],
                     'WEIGHT': [0.5, 0.2, 0.4],
                     'REWEIGHT': [0.5 * (3/20), 0.2 * (3/20), 0.4 * (3/20)]}
    expected_2013 = {'YEAR': [2013], 'WEIGHT': [1.5],
                     'REWEIGHT': [1.5 * (1/20)]}
    expected_2014 = {'YEAR': [2014, 2014], 'WEIGHT': [1.6, 0.8],
                     'REWEIGHT': [1.6 * (2/20), 0.8 * (2/20)]}
    expected_2015 = {'YEAR': [2015, 2015, 2015, 2015, 2015],
                     'WEIGHT': [3.4, 4.8, 5.0, 6.1, 3.4],
                     'REWEIGHT': [3.4 * (5/20), 4.8 * (5/20), 5.0 * (5/20),
                                  6.1 * (5/20), 3.4 * (5/20)]}
    expected_2016 = {'YEAR': [2016, 2016], 'WEIGHT': [4.1, 5.3],
                     'REWEIGHT': [4.1 * (2/20), 5.3 * (2/20)]}
    expected_2017 = {'YEAR': [2017], 'WEIGHT': [0.4],
                     'REWEIGHT': [0.4 * (1/20)]}
    expected_2018 = {'YEAR': [2018, 2018, 2018, 2018],
                     'WEIGHT': [0.9, 7.5, 5.5, 1.0],
                     'REWEIGHT': [0.9 * (4/20), 7.5 * (4/20),
                                  5.5 * (4/20), 1.0 * (4/20)]}
    expected_2019 = {'YEAR': [2019, 2019], 'WEIGHT': [2.2, 0.5],
                     'REWEIGHT': [2.2 * (2/20), 0.5 * (2/20)]}
    test_df_one = pd.DataFrame(test_2012)
    test_df_two = pd.DataFrame(test_2013)
    test_df_three = pd.DataFrame(test_2014)
    test_df_four = pd.DataFrame(test_2015)
    test_df_five = pd.DataFrame(test_2016)
    test_df_six = pd.DataFrame(test_2017)
    test_df_seven = pd.DataFrame(test_2018)
    test_df_eight = pd.DataFrame(test_2019)
    all_test = [test_df_one, test_df_two, test_df_three, test_df_four,
                test_df_five, test_df_six, test_df_seven, test_df_eight]
    expected_one = pd.DataFrame(expected_2012)
    expected_two = pd.DataFrame(expected_2013)
    expected_three = pd.DataFrame(expected_2014)
    expected_four = pd.DataFrame(expected_2015)
    expected_five = pd.DataFrame(expected_2016)
    expected_six = pd.DataFrame(expected_2017)
    expected_seven = pd.DataFrame(expected_2018)
    expected_eight = pd.DataFrame(expected_2019)
    all_expected = [expected_one, expected_two, expected_three, expected_four,
                    expected_five, expected_six, expected_seven,
                    expected_eight]
    actual_result = weight_state_data(all_test)
    for i in range(len(actual_result)):
        result = actual_result[i]
        expected = all_expected[i]
        assert_frame_equal(result, expected)
    print('passed reweighting tests!')


def test_additional_vars():
    """
    Tests the construct_additional_vars method.
    """
    test_one = {'STATE': [2, 2, 41, 27, 23, 24, 55, 4, 4],
                'YEAR': [2017, 2018, 2018, 2019, 2019, 2015, 2017, 2019, 2017],
                'NUMADULT': [4, 9, 7, 6, 2, 5, 3, 2, 1],
                'GENHLTH': [5, 4, 3, 4, 1, 1, 2, 4, 2],
                'PHYSHLTH': [1, 20, 15, 16, 18, 16, 19, 20, 17],
                'MENTHLTH': [29, 2, 21, 8, 5, 3, 9, 10, 11],
                'POORHLTH': [7, 9, 17, 23, 21, 14, 1, 0, 5],
                'HLTHPLN': [1, 1, 1, 2, 2, 2, 2, 1, 1],
                'PERSDOC': [1, 3, 2, 3, 2, 1, 3, 2, 1],
                'MEDCOST': [1, 1, 2, 1, 2, 2, 1, 1, 1],
                'CHECKUP': [2, 1, 4, 1, 8, 8, 4, 2, 1],
                'HRTDIS': [2, 2, 1, 2, 2, 2, 2, 2, 1],
                'HRTATTCK': [2, 2, 2, 2, 2, 1, 2, 2, 2],
                'STROKE': [2, 2, 2, 2, 2, 2, 2, 2, 1],
                'SKNCNCR': [2, 2, 2, 2, 2, 1, 2, 1, 1],
                'OTHERCNCR': [2, 2, 2, 2, 2, 1, 2, 2, 2],
                'MARITAL': [1, 1, 1, 2, 2, 1, 2, 2, 1],
                'EDUCA': [4, 5, 1, 3, 3, 3, 5, 1, 2],
                'RENT': [1, 1, 2, 2, 3, 3, 2, 1, 1],
                'CHILDREN': [2, 1, 7, 10, 5, 1, 2, 1, 4],
                'INCOME': [2, 3, 1, 6, 7, 5, 4, 6, 5],
                'AGE': [2, 1, 2, 2, 1, 1, 1, 2, 2],
                'BMI': [1, 1, 1, 1, 2, 2, 2, 1, 1],
                'WEIGHT': [9, 7, 99, 10, 5, 5, 9, 7, 10],
                'REWEIGHT': [2, 4, 5, 7, 8, 5, 3, 8.5, 7],
                'COPD': [2, 2, 2, 2, 2, 2, 2, 2, 2],
                'ARTH': [2, 2, 2, 2, 2, 2, 2, 1, 1],
                'KIDDIS': [2, 2, 2, 2, 2, 2, 2, 2, 1],
                'DIABETE': [3, 2, 3, 4, 4, 4, 3, 2, 1],
                'SEX': [1, 1, 2, 2, 1, 1, 1, 2, 2],
                'EMPLOY': [8, 7, 1, 2, 3, 3, 7, 8, 1],
                'RACE': [1, 2, 3, 4, 5, 5, 4, 3, 1],
                'DEPRESS': [2, 2, 2, 2, 2, 2, 2, 1, 2]}

    expected_one = {'STATE': [2, 2, 41, 27, 23, 24, 55, 4, 4],
                    'YEAR': [2017, 2018, 2018, 2019, 2019, 2015, 2017, 2019,
                             2017],
                    'NUMADULT': [4, 9, 7, 6, 2, 5, 3, 2, 1],
                    'GENHLTH': [5, 4, 3, 4, 1, 1, 2, 4, 2],
                    'PHYSHLTH': [1, 20, 15, 16, 18, 16, 19, 20, 17],
                    'MENTHLTH': [29, 2, 21, 8, 5, 3, 9, 10, 11],
                    'POORHLTH': [7, 9, 17, 23, 21, 14, 1, 0, 5],
                    'HLTHPLN': [1, 1, 1, 2, 2, 2, 2, 1, 1],
                    'PERSDOC': [1, 3, 2, 3, 2, 1, 3, 2, 1],
                    'MEDCOST': [1, 1, 2, 1, 2, 2, 1, 1, 1],
                    'CHECKUP': [2, 1, 4, 1, 8, 8, 4, 2, 1],
                    'HRTDIS': [2, 2, 1, 2, 2, 2, 2, 2, 1],
                    'HRTATTCK': [2, 2, 2, 2, 2, 1, 2, 2, 2],
                    'STROKE': [2, 2, 2, 2, 2, 2, 2, 2, 1],
                    'SKNCNCR': [2, 2, 2, 2, 2, 1, 2, 1, 1],
                    'OTHERCNCR': [2, 2, 2, 2, 2, 1, 2, 2, 2],
                    'MARITAL': [1, 1, 1, 2, 2, 1, 2, 2, 1],
                    'EDUCA': [4, 5, 1, 3, 3, 3, 5, 1, 2],
                    'RENT': [1, 1, 2, 2, 3, 3, 2, 1, 1],
                    'CHILDREN': [2, 1, 7, 10, 5, 1, 2, 1, 4],
                    'INCOME': [2, 3, 1, 6, 7, 5, 4, 6, 5],
                    'AGE': [2, 1, 2, 2, 1, 1, 1, 2, 2],
                    'BMI': [1, 1, 1, 1, 2, 2, 2, 1, 1],
                    'WEIGHT': [9, 7, 99, 10, 5, 5, 9, 7, 10],
                    'REWEIGHT': [2, 4, 5, 7, 8, 5, 3, 8.5, 7],
                    'COPD': [2, 2, 2, 2, 2, 2, 2, 2, 2],
                    'ARTH': [2, 2, 2, 2, 2, 2, 2, 1, 1],
                    'KIDDIS': [2, 2, 2, 2, 2, 2, 2, 2, 1],
                    'DIABETE': [3, 2, 3, 4, 4, 4, 3, 2, 1],
                    'SEX': [1, 1, 2, 2, 1, 1, 1, 2, 2],
                    'EMPLOY': [8, 7, 1, 2, 3, 3, 7, 8, 1],
                    'RACE': [1, 2, 3, 4, 5, 5, 4, 3, 1],
                    'DEPRESS': [2, 2, 2, 2, 2, 2, 2, 1, 2],
                    'TREAT_EARLY': [1, 1, 1, 1, 0, 0, 0, 0, 0],
                    'TREAT_LATE': [0, 0, 0, 0, 1, 1, 1, 0, 0],
                    'TIME_EARLY': [0, 1, 1, 1, 1, 0, 0, 1, 0],
                    'TIME_LATE': [0, 0, 0, 1, 1, 0, 0, 1, 0],
                    'CHR_DIS': [0, 1, 1, 0, 1, 1, 1, 1, 1]}
    test_df = pd.DataFrame(test_one)
    expected_df = pd.DataFrame(expected_one)
    actual_df = construct_additional_vars(test_df)
    assert_frame_equal(actual_df, expected_df)


def test_proportions_prep():
    """
    Tests the proportions_prep_for_outcomes method.
    """
    test_one = {'STATE': [2, 2, 23, 23, 4],
                'GENHLTH': [5, 4, 3, 4, 1],
                'PHYSHLTH': [1, 20, 15, 16, 18],
                'MENTHLTH': [29, 2, 21, 8, 5],
                'POORHLTH': [7, 9, 17, 23, 21],
                'PERSDOC': [1, 3, 2, 3, 2],
                'CHECKUP': [2, 1, 4, 1, 8]}

    expected_one = {'STATE': [2, 2, 23, 23, 4],
                    'GENHLTH': [5, 4, 3, 4, 1],
                    'PHYSHLTH': [1, 20, 15, 16, 18],
                    'MENTHLTH': [29, 2, 21, 8, 5],
                    'POORHLTH': [7, 9, 17, 23, 21],
                    'PERSDOC': [1, 3, 2, 3, 2],
                    'CHECKUP': [2, 1, 4, 1, 8],
                    'GOOD_GENHLTH': [0, 0, 1, 0, 1],
                    'PHYS_DISTRESS': [0, 1, 1, 1, 1],
                    'MENT_DISTRESS': [1, 0, 1, 0, 0],
                    'POOR_OVR_HLTH': [0, 0, 1, 1, 1],
                    'HAS_PERSDOC': [1, 0, 1, 0, 1],
                    'ANNUAL_CHECKUP': [0, 1, 0, 1, 0]}

    test_df = pd.DataFrame(test_one)
    expected_df = pd.DataFrame(expected_one)
    actual_df = proportions_prep_for_outcomes(test_df)
    assert_frame_equal(actual_df, expected_df)


def test_get_proportions_data():
    """
    Tests the get_proportions_data method.
    """
    test_data = {'YEAR': [2012, 2012, 2012, 2012, 2013, 2013, 2013, 2013,
                          2014, 2014, 2014, 2014, 2015, 2015, 2015, 2015,
                          2016, 2016, 2016, 2016, 2017, 2017, 2017, 2017,
                          2018, 2018, 2018, 2018],
                 'TREAT_EARLY': [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1,
                                 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0],
                 'TREAT_LATE': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0,
                                1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
                 'GOOD_GENHLTH': [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
                                  0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
                 'PHYS_DISTRESS': [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                                   1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
                 'MENT_DISTRESS': [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1,
                                   1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
                 'POOR_OVR_HLTH': [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                                   1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
                 'HLTHPLN': [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1,
                             1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
                 'HAS_PERSDOC': [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1,
                                 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1],
                 'MEDCOST': [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1,
                             1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
                 'ANNUAL_CHECKUP': [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
                                    0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]}

    expected_data = {'YEAR': [2012, 2012, 2012, 2012, 2013, 2013, 2013, 2013,
                              2014, 2014, 2014, 2014, 2015, 2015, 2015, 2015,
                              2016, 2016, 2016, 2016, 2017, 2017, 2017, 2017,
                              2018, 2018, 2018, 2018],
                     'TREAT_EARLY': [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0,
                                     1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0],
                     'TREAT_LATE': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1,
                                    0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
                     'GOOD_GENHLTH': [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0,
                                      1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0,
                                      0, 1],
                     'PHYS_DISTRESS': [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1,
                                       0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0,
                                       0, 1],
                     'MENT_DISTRESS': [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1,
                                       1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1,
                                       0, 0],
                     'POOR_OVR_HLTH': [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1,
                                       0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0,
                                       0, 1],
                     'HLTHPLN': [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1,
                                 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
                     'HAS_PERSDOC': [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
                                     1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1],
                     'MEDCOST': [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0,
                                 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
                     'ANNUAL_CHECKUP': [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0,
                                        1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0,
                                        1, 0],
                     'PROP_GOOD_GENHLTH': [1/3, 1/1, 1/3, 1/3, 0/2, 1/1, 1/1,
                                           0/2, 1/2, 1/2, 0/1, 1/1, 1/2, 1/2,
                                           1/2, 1/2, 1/1, 1/2, 1/2, 1/1, 1/2,
                                           1/2, 1/2, 1/2, 1/2, 1/2, 0/1, 1/1],
                     'PROP_PHYS_DISTRESS': [1/3, 0/1, 1/3, 1/3, 1/2, 0/1, 0/1,
                                            1/2, 1/2, 1/2, 0/1, 1/1, 1/2, 1/2,
                                            1/2, 1/2, 1/1, 1/2, 1/2, 0/1, 2/2,
                                            1/2, 1/2, 2/2, 0/2, 0/2, 0/1, 1/1],
                     'PROP_MENT_DISTRESS': [2/3, 0/1, 2/3, 2/3, 0/2, 1/1, 1/1,
                                            0/2, 0/2, 0/2, 1/1, 0/1, 2/2, 2/2,
                                            2/2, 2/2, 0/1, 0/2, 0/2, 1/1, 2/2,
                                            0/2, 0/2, 2/2, 2/2, 2/2, 0/1, 0/1],
                     'PROP_POOR_OVR_HLTH': [2/3, 1/1, 2/3, 2/3, 1/2, 0/1, 0/1,
                                            1/2, 1/2, 1/2, 0/1, 0/1, 1/2, 1/2,
                                            1/2, 1/2, 1/1, 1/2, 1/2, 0/1, 1/2,
                                            1/2, 1/2, 1/2, 1/2, 1/2, 0/1, 1/1],
                     'PROP_HLTHPLN': [2/3, 0/1, 2/3, 2/3, 2/2, 1/1, 0/1, 2/2,
                                      1/2, 1/2, 1/1, 1/1, 1/2, 1/2, 1/2, 1/2,
                                      1/1, 1/2, 1/2, 0/1, 0/2, 2/2, 2/2, 0/2,
                                      2/2, 2/2, 0/1, 0/1],
                     'PROP_HAS_PERSDOC': [3/3, 0/1, 3/3, 3/3, 2/2, 1/1, 0/1,
                                          2/2, 1/2, 1/2, 1/1, 1/1, 2/2, 1/2,
                                          2/2, 1/2, 0/1, 1/2, 1/2, 1/1, 0/2,
                                          2/2, 2/2, 0/2, 0/2, 0/2, 1/1, 1/1],
                     'PROP_MEDCOST': [2/3, 1/1, 2/3, 2/3, 1/2, 0/1, 1/1, 1/2,
                                      1/2, 1/2, 0/1, 0/1, 1/2, 2/2, 1/2, 2/2,
                                      1/1, 0/2, 0/2, 1/1, 1/2, 1/2, 1/2, 1/2,
                                      2/2, 2/2, 0/1, 0/1],
                     'PROP_ANNUAL_CHECKUP': [1/3, 0/1, 1/3, 1/3, 1/2, 1/1,
                                             0/1, 1/2, 1/2, 1/2, 1/1, 0/1,
                                             0/2, 2/2, 0/2, 2/2, 1/1, 0/2,
                                             0/2, 2/2, 1/2, 1/2, 1/2, 1/2,
                                             1/2, 1/2, 1/1, 0/1]}

    test_df = pd.DataFrame(test_data)
    expected_df = pd.DataFrame(expected_data)
    actual_df = get_proportions_data(test_df)
    actual_df = actual_df.drop(columns=['TREAT', 'ALL_PROP_GOOD_GENHLTH',
                                        'ALL_PROP_PHYS_DISTRESS',
                                        'ALL_PROP_MENT_DISTRESS',
                                        'ALL_PROP_POOR_OVR_HLTH',
                                        'ALL_PROP_HLTHPLN',
                                        'ALL_PROP_HAS_PERSDOC',
                                        'ALL_PROP_MEDCOST',
                                        'ALL_PROP_ANNUAL_CHECKUP'])
    assert_frame_equal(expected_df, actual_df)


def test_binary_recode():
    """
    Tests the binary_recode method from analysis.py
    """
    test_one = {'Unnamed: 0': [0, 1, 2, 3, 4],
                'STATE': [2, 2, 23, 23, 4],
                'YEAR': [2017, 2018, 2018, 2019, 2019],
                'NUMADULT': [4, 9, 7, 6, 2],
                'GENHLTH': [5, 4, 3, 4, 1],
                'PHYSHLTH': [1, 2, 15, 5, 18],
                'MENTHLTH': [7, 2, 21, 8, 5],
                'POORHLTH': [7, 9, 17, 23, 8],
                'HLTHPLN': [1, 1, 1, 2, 2],
                'PERSDOC': [1, 3, 2, 3, 2],
                'MEDCOST': [1, 1, 2, 1, 2],
                'CHECKUP': [2, 1, 4, 1, 8],
                'HRTDIS': [2, 2, 1, 2, 2],
                'HRTATTCK': [2, 2, 2, 2, 2],
                'STROKE': [2, 2, 2, 2, 2],
                'SKNCNCR': [2, 2, 2, 2, 2],
                'OTHERCNCR': [2, 2, 2, 2, 2],
                'MARITAL': [1, 1, 1, 2, 2],
                'EDUCA': [4, 5, 1, 3, 3],
                'RENT': [1, 1, 2, 2, 3],
                'CHILDREN': [2, 1, 7, 8, 11],
                'INCOME': [2, 3, 1, 6, 7],
                'AGE': [2, 1, 2, 2, 1],
                'BMI': [1, 1, 1, 1, 2],
                'WEIGHT': [9, 7, 99, 10, 5],
                'REWEIGHT': [2, 4, 5, 7, 8],
                'COPD': [2, 2, 2, 2, 2],
                'ARTH': [2, 2, 2, 2, 2],
                'KIDDIS': [2, 2, 2, 2, 2],
                'DIABETE': [3, 2, 3, 4, 4],
                'SEX': [1, 1, 2, 2, 1],
                'EMPLOY': [8, 7, 1, 2, 3],
                'RACE': [1, 2, 3, 4, 5],
                'DEPRESS': [2, 2, 2, 2, 2]}

    expected_one = {'STATE': [2, 2, 23, 23, 4],
                    'YEAR': [2017, 2018, 2018, 2019, 2019],
                    'NUMADULT': [4, 9, 7, 6, 2],
                    'GENHLTH': [5, 4, 3, 4, 1],
                    'PHYSHLTH': [1, 2, 15, 5, 18],
                    'MENTHLTH': [7, 2, 21, 8, 5],
                    'POORHLTH': [7, 9, 17, 23, 8],
                    'HLTHPLN': [1, 1, 1, 0, 0],
                    'PERSDOC': [1, 3, 2, 3, 2],
                    'MEDCOST': [1, 1, 0, 1, 0],
                    'CHECKUP': [2, 1, 4, 1, 8],
                    'HRTDIS': [0, 0, 1, 0, 0],
                    'HRTATTCK': [0, 0, 0, 0, 0],
                    'STROKE': [0, 0, 0, 0, 0],
                    'SKNCNCR': [0, 0, 0, 0, 0],
                    'OTHERCNCR': [0, 0, 0, 0, 0],
                    'MARITAL': [1, 1, 1, 2, 2],
                    'EDUCA': [4, 5, 1, 3, 3],
                    'RENT': [1, 1, 2, 2, 3],
                    'CHILDREN': [2, 1, 7, 8, 11],
                    'INCOME': [2, 3, 1, 6, 7],
                    'AGE': [2, 1, 2, 2, 1],
                    'BMI': [0, 0, 0, 0, 1],
                    'WEIGHT': [9, 7, 99, 10, 5],
                    'REWEIGHT': [2, 4, 5, 7, 8],
                    'COPD': [0, 0, 0, 0, 0],
                    'ARTH': [0, 0, 0, 0, 0],
                    'KIDDIS': [0, 0, 0, 0, 0],
                    'DIABETE': [3, 2, 3, 4, 4],
                    'SEX': [1, 1, 2, 2, 1],
                    'EMPLOY': [8, 7, 1, 2, 3],
                    'RACE': [1, 2, 3, 4, 5],
                    'DEPRESS': [0, 0, 0, 0, 0]}
    test_df = pd.DataFrame(test_one)
    expected_df = pd.DataFrame(expected_one)
    actual_df = binary_recode(test_df)
    assert_frame_equal(expected_df, actual_df)


def main():
    # Testing for simplify.py
    print('Starting testing for simplify.py:')
    current_wd = os.getcwd()
    test_rewrite_data(current_wd)
    test_recode()
    test_get_state_obs()
    test_weight_state_data()
    test_additional_vars()
    test_binary_recode()
    test_proportions_prep()
    test_get_proportions_data()
    print('All simplify.py tests passed!')


if __name__ == '__main__':
    main()
