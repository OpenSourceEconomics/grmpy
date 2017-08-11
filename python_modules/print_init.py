def print_dict(dict_, file_name):
    """Creates an init file from a given dictionary"""

    labels = ['SIMULATION', 'TREATED', 'UNTREATED', 'COST', 'DIST']



    with open(file_name, 'w') as file_:

        for label in labels:

            file_.write(label + '\n\n')

            if label == 'SIMULATION':

                structure = ['agents', 'seed', 'source']

                for key_ in structure:

                    if key_ == 'source':
                        str_ = '{0:<25} {1:20}\n'

                        file_.write(str_.format(key_, dict_[label][key_]))
                    else:
                        str_ = '{0:<10} {1:20}\n'

                        file_.write(str_.format(
                            key_, dict_['SIMULATION'][key_]))

            elif label in ['TREATED', 'UNTREATED', 'COST', 'DIST']:
                key_ = 'coeff'

                for i in range(len(dict_[label][key_])):
                    str_ = '{0:<10} {1:20.4f}\n'
                    file_.write(str_.format(key_, dict_[label]['coeff'][i]))

            file_.write('\n')
