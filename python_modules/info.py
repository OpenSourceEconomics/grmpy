
import numpy as np
import pandas as pd

def _collect_information(data_frame):
    '''Calculates the required information for the info file'''
    # Number of individuals:
    Indiv = len(data_frame)

    # Counts by treatment status
    treatment_count = data_frame[data_frame.D == 1].count()['D']
    untreated_count= Indiv - treatment_count
    # Average Treatment Effect
    ATE = np.mean(data_frame.Y1 - data_frame.Y0)
    # Treatment on Treated
    TT = np.mean(data_frame.Y1[data_frame.D == 1])
    - np.mean(data_frame.Y0[data_frame.D == 1])
    # Treatment on Untreated
    TUT = np.mean(data_frame.Y0[data_frame.D == 1])
    - np.mean(data_frame.Y0[data_frame.D == 0])
    # Average observed wage overall and by treatment status
    Mean = np.mean(data_frame.Y)
    Mean_treat = np.mean(data_frame.Y[data_frame.D == 1])
    Mean_untreat = np.mean(data_frame.Y[data_frame.D == 0])

    data = {
        'Number of Agents': Indiv, 'Treated Agents': treatment_count, 'Untreated Agents': untreated_count, 'Average Treatment Effect': ATE, 'Treatment on Treated': TT,
        'Treatment on Untreated': TUT, 'Mean': Mean, 'Mean Treated': Mean_treat, 'Mean Untreated': Mean_untreat
    }


    return data





def _print_info(data_frame, coeffs, file_name):
    '''Prints an info file for the specififc dataset'''

    data_ = _collect_information(data_frame)

    labels= ['Simulation', 'Additional Information', 'Effects', 'Model Paramerization']

    with open(file_name + '.grmpy.info', 'w') as file_:

        for label in labels:
            file_.write('\n' + label + '\n\n')

            if label == 'Simulation':
                structure= ['Number of Agents', 'Treated Agents', 'Untreated Agents']

                for s in structure:
                    str_ = '{0:<25} {1:20}\n'

                    file_.write(str_.format(s+':', data_[s]))


            elif label =='Additional Information':
                structure = ['Mean', 'Mean Treated', 'Mean Untreated']

                for s in structure:
                    str_ = '{0:<25} {1:20.4f}\n'
                    file_.write(str_.format(s+ ':', data_[s]))

            elif label == 'Effects':
                structure = ['Average Treatment Effect', 'Treatment on Treated', 'Treatment on Untreated']

                for s in structure:
                    str_ = '{0:<25} {1:20.4f}\n'
                    file_.write(str_.format(s +':' , data_[s]))

            else:
                structure =['Treated Coeff', 'Untreated Coeff', 'Cost Coeff']


                for s in structure:
                    file_.write('\n' + s + '\n\n')

                    if s == 'Treated Coeff' :
                        for i in range(len(coeffs[0])):
                            str_ = '{0:<25} {1:20.4f}\n'
                            file_.write(str_.format('coeff', coeffs[0][i]))
                    elif s == 'Untreated Coeff':
                        for i in range(len(coeffs[1])):
                            str_ = '{0:<25} {1:20.4f}\n'
                            file_.write(str_.format('coeff', coeffs[1][i]))
                    else:
                        for i in range(len(coeffs[2])):
                            str_ = '{0:<25} {1:20.4f}\n'
                            file_.write(str_.format('coeff', coeffs[2][i]))
