

import numpy as np
import pandas as pd
import pickle


def _write_output(end, exog, err, source, is_deterministic):
    '''Converts simulated data to a panda data frame
    and saves the data in an html file/pickle'''
    column = ['Y', 'D']

    # Stack arrays
    if not is_deterministic:
        data = np.column_stack(
            (end[0], end[1], exog[0], exog[1], end[2], end[3]))
        data = np.column_stack(
            (data, err[0][0:, 0], err[0][0:, 1], err[0][0:, 2], err[1]))
        # List of column names

        for i in range(exog[0].shape[1]):
            str_ = 'X_' + str(i)
            column.append(str_)
        for i in range(exog[1].shape[1]):
            str_ = 'Z_' + str(i)
            column.append(str_)
    else:
        data = np.column_stack((end[0], end[1], end[2], end[3], err[0][0:, 0], err[0][0:, 1], err[0][0:, 2], err[1]))
    column = column + ['Y1', 'Y0', 'U0', 'U1', 'UC', 'V']
    # Generate data frame, save it with pickle and create a html file

    df=pd.DataFrame(data=data, columns=column)

    df.to_pickle(source + '.grmpy.pkl')

    with open(source + '.grmpy.txt', 'w') as file_:
        df.to_string(file_, index=False, header=True, na_rep='.', col_space=15)

    return df
