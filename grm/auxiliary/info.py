
import numpy as np
import pandas as pd


def _collect_information(data_frame):
    '''Calculates the required information for the info file'''
    # Number of individuals:
    Indiv = len(data_frame)

    # Counts by treatment status
    treated_num = data_frame[data_frame.D == 1].count()['D']
    untreated_num = Indiv - treated_num

    no_treatment, all_treatment = _adjust_collecting(
        treated_num, untreated_num)
    # Average Treatment Effect
    ATE = np.mean(data_frame.Y1 - data_frame.Y0)
    Mean = np.mean(data_frame.Y)

    TT, TUT, Mean_treat, SD_treat, Quant_treat, Mean_untreat, SD_untreat, Quant_untreat = _calc_parameters(data_frame, no_treatment, all_treatment)

    data = {
        'Number of Agents': Indiv, 'Treated Agents': treated_num,
        'Untreated Agents': untreated_num, 'Average Treatment Effect': ATE,
        'Treatment on Treated': TT, 'Treatment on Untreated': TUT,
        'Mean': Mean, 'Mean Treated': Mean_treat, 'Mean Untreated': Mean_untreat,
        'Std Treated': SD_treat, 'Std Untreated': SD_untreat, 'Quantiles Treated': Quant_treat, 'Quantiles Untreated': Quant_untreat
    }

    return data


def _print_info(data_frame, coeffs, file_name):

    data_ = _collect_information(data_frame)
    no_treatment, all_treatment = _adjust_collecting(data_['Treated Agents'], data_['Untreated Agents'])
    '''Prints an info file for the specififc dataset'''
    labels = ['Simulation', 'Additional Information', 'Effects',
    'Model Paramerization']

    with open(file_name + '.grmpy.info', 'w') as file_:

        for label in labels:
            file_.write('\n' + label + '\n\n')

            if label == 'Simulation':
                structure = [
                'Number of Agents', 'Treated Agents', 'Untreated Agents'
                ]

                for s in structure:
                    str_ = '{0:<25} {1:20}\n'

                    file_.write(str_.format(s + ':', data_[s]))

            elif label == 'Additional Information':
                if no_treatment:
                    _print_notreated(data_, file_)
                elif all_treatment:
                    _print_alltreated(data_, file_)
                else:
                    _print_normal(data_, file_)


            elif label == 'Effects':
                structure = [
                'Average Treatment Effect', 'Treatment on Treated', 'Treatment on Untreated'
                ]

                for s in structure:
                    if s == 'Treatment on Treated':
                        if no_treatment:
                            str_ = '{0:<25} {1:>20}\n'
                            file_.write(str_.format(s + ':', data_[s]))
                        else:
                            str_ = '{0:<25} {1:20.4f}\n'
                            file_.write(str_.format(s + ':', data_[s]))
                    elif s == 'Treatment on Untreated':
                        if all_treatment:
                            str_ = '{0:<25} {1:>20}\n'
                            file_.write(str_.format(s + ':', data_[s]))
                        else:
                            str_ = '{0:<25} {1:20.4f}\n'
                            file_.write(str_.format(s + ':', data_[s]))
                    else:
                        str_ = '{0:<25} {1:20.4f}\n'
                        file_.write(str_.format(s + ':', data_[s]))

            else:
                structure = [
                    'TREATED', 'UNTREATED', 'COST', 'DIST'
                ]

                for s in structure:
                    file_.write('\n' + s + '\n\n')

                    if s == 'TREATED':
                        str_ = '{0:<25}'
                        str_coeff = ''
                        for i in range(len(coeffs[1])):
                            str_coeff = str_coeff + \
                                ' {0:10.4f}'.format(coeffs[1][i])
                        file_.write(str_.format('coeff') + str_coeff + '\n')
                    elif s == 'UNTREATED':
                        str_ = '{0:<25}'
                        str_coeff = ''
                        for i in range(len(coeffs[0])):
                            str_coeff = str_coeff + \
                                ' {0:10.4f}'.format(coeffs[0][i])
                        file_.write(str_.format('coeff') + str_coeff + '\n')
                    elif s == 'COST':
                        str_ = '{0:<25}'
                        str_coeff = ''
                        for i in range(len(coeffs[2])):
                            str_coeff = str_coeff + \
                                ' {0:10.4f}'.format(coeffs[2][i])
                        file_.write(str_.format('coeff') + str_coeff + '\n')
                    else:
                        str_ = '{0:<25}'
                        str_coeff = ''
                        for i in range(len(coeffs[3])):
                            str_coeff = str_coeff + \
                                ' {0:10.4f}'.format(coeffs[3][i])
                        file_.write(str_.format('coeff') + str_coeff + '\n')


def _print_notreated(data_, file_name):



    structure = ['Mean', 'Mean Treated', 'Std Treated', 'Quantiles Treated',
        'Mean Untreated', 'Std Untreated', 'Quantiles Untreated'
        ]
    for s in structure:
        if s == 'Quantiles Untreated':
            str_ = '{0:<25} {1:20.4f} {2:10.4f} {3:10.4f} \n\n'
            q02 = data_[s].loc[0.2]
            q05 = data_[s].loc[0.5]
            q08 = data_[s].loc[0.8]
            file_name.write(str_.format(s + ':', q02, q05, q08))
        else:
            if isinstance(data_[s], str):
                str_ = '{0:<25} {1:>20}\n'
                file_name.write(str_.format(s + ':', data_[s]))
            else:
                str_ = '{0:<25} {1:20.4f}\n'
                file_name.write(str_.format(s + ':', data_[s]))







def _print_alltreated(data_, file_name):


    structure = ['Mean', 'Mean Treated', 'Std Treated', 'Quantiles Treated',
        'Mean Untreated', 'Std Untreated', 'Quantiles Untreated'
        ]
    for s in structure:
        if s == 'Quantiles Treated':
            str_ = '{0:<25} {1:20.4f} {2:10.4f} {3:10.4f} \n\n'
            q02 = data_[s].loc[0.2]
            q05 = data_[s].loc[0.5]
            q08 = data_[s].loc[0.8]
            file_name.write(str_.format(s + ':', q02, q05, q08))
        else:
            if isinstance(data_[s], str):
                str_ = '{0:<25} {1:>20}\n'
                file_name.write(str_.format(s + ':', data_[s]))
            else:
                str_ = '{0:<25} {1:20.4f}\n'
                file_name.write(str_.format(s + ':', data_[s]))




def _print_normal(data_, file_name):
    structure = ['Mean', 'Mean Treated', 'Std Treated', 'Quantiles Treated',
        'Mean Untreated', 'Std Untreated', 'Quantiles Untreated'
        ]
    for s in structure:
        if s.startswith('Quantiles'):
            str_ = '{0:<25} {1:20.4f} {2:10.4f} {3:10.4f} \n\n'
            q02 = data_[s].loc[0.2]
            q05 = data_[s].loc[0.5]
            q08 = data_[s].loc[0.8]
            file_name.write(str_.format(s + ':', q02, q05, q08))
        elif s == 'Mean':
            str_ = '{0:<25} {1:20.4f}\n\n'
            file_name.write(str_.format(s + ':', data_[s]))

        else:
            str_ = '{0:<25} {1:20.4f}\n'
            file_name.write(str_.format(s + ':', data_[s]))





def _calc_parameters(data_frame, no_treatment, all_treatment):

    if not no_treatment and not all_treatment:

        # Treatment on the Treated
        TT = np.mean(data_frame.Y1[data_frame.D == 1])
        - np.mean(data_frame.Y0[data_frame.D == 1])
        # Treatment on Untreated
        TUT = np.mean(data_frame.Y1[data_frame.D == 0])
        - np.mean(data_frame.Y0[data_frame.D == 0])
        # Average observed wage overall and by treatment status
        Mean_treat = np.mean(data_frame.Y[data_frame.D == 1])
        SD_treat = np.std(data_frame.Y[data_frame.D == 1])
        Quant_treat = data_frame.Y[data_frame.D == 1].quantile([0.2, 0.5, 0.8])
        Mean_untreat = np.mean(data_frame.Y[data_frame.D == 0])
        Quant_untreat = data_frame.Y[data_frame.D == 0].quantile([
                                                                 0.2, 0.5, 0.8])
        SD_untreat = np.std(data_frame.Y[data_frame.D == 0])
        # Print out model parameterization

    elif no_treatment:
        TT = "---"
        TUT = np.mean(data_frame.Y1[data_frame.D == 0])
        - np.mean(data_frame.Y0[data_frame.D == 0])
        Mean_treat = "---"
        SD_treat = "---"
        Quant_treat = "---"
        Mean_untreat = np.mean(data_frame.Y[data_frame.D == 0])
        Quant_untreat = data_frame.Y[data_frame.D == 0].quantile([0.2, 0.5, 0.8])
        SD_untreat = np.std(data_frame.Y[data_frame.D == 0])



    elif all_treatment:
        TT = np.mean(data_frame.Y1[data_frame.D == 1])
        - np.mean(data_frame.Y0[data_frame.D == 1])
        TUT = "---"
        Mean_untreat = "---"
        Quant_untreat = "---"
        SD_untreat = "---"
        Mean_treat = np.mean(data_frame.Y[data_frame.D == 1])
        SD_treat = np.std(data_frame.Y[data_frame.D == 1])
        Quant_treat = data_frame.Y[data_frame.D == 1].quantile([0.2, 0.5, 0.8])

    else:
        print('Error,')
    return TT, TUT, Mean_treat, SD_treat, Quant_treat, Mean_untreat, SD_untreat, Quant_untreat


def _adjust_collecting(treated_num, untreated_num):

    if treated_num == 0:
        no_treatment = True
    else:
        no_treatment = False


    if untreated_num == 0:
        all_treatment = True
    else:
        all_treatment = False

    assert no_treatment != all_treatment or no_treatment == all_treatment == False


    return no_treatment, all_treatment
