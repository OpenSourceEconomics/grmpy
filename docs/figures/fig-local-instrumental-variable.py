import pickle as pkl
import numpy as np
import copy 
import linecache
import shlex

from statsmodels.sandbox.regression.gmm import IV2SLS
from fig_auxiliary_functions import *
import statsmodels.api as sm
import seaborn.apionly as sns

from grmpy.estimate.estimate import estimate
from grmpy.simulate.simulate import simulate

def update_correlation_structure(model_dict, rho):
    """This function takes a valid model specification and updates the correlation structure
    among the unobservables."""
    # We first extract the baseline information from the model dictionary.
    sd_v = model_dict['DIST']['all'][-1]
    sd_u =  model_dict['DIST']['all'][0]
    
    # Now we construct the implied covariance, which is relevant for the initialization file.
    cov = rho * sd_v * sd_u
    model_dict['DIST']['all'][2] = cov
    # We print out the specification to an initialization file with the name mc_init.grmpy.ini.
    print_model_dict(model_dict)
    
def collect_effects(model_base, which, grid_points):
    """This function collects numerous effects for alternative correlation structures."""
    model_mc = copy.deepcopy(model_base)
    
    effects = []
    for rho in np.linspace(0.00, 0.99, grid_points):
        
        # We create a new initialization file with an  updated correlation structure.
        update_correlation_structure(model_mc, rho)
    
        # We use this new file to simulate a new sample.
        df_mc = simulate('mc_init.grmpy.ini')
        
        # We extract auxiliary objects for further processing.
        endog, exog, instr = df_mc['Y'], df_mc[['X_0', 'D']], df_mc[['X_0', 'Z_1']]
        d_treated = df_mc['D'] == 1
                                                                    
        # We calculate our parameter of interest.
        label = which.lower()
        if label == 'local_instrumental_variables':
            estimate('mc_init.grmpy.ini')
            stat = get_effect_grmpy()
        else:
            raise NotImplementedError
        
        effects += [stat]
    
    return effects

model_base = get_model_dict('mc_exploration.grmpy.ini')
model_base['SIMULATION']['source'] = 'mc_data'
          
df_base = simulate('mc_exploration.grmpy.ini')
df_base.head()


ate = np.mean(df_base['Y1'] - df_base['Y0'])
true_effect = ate

effects = collect_effects(model_base, 'local_instrumental_variables', 10)
plot_estimates(true_effect, effects)
plt.savefig('fig-local-instrumental-variable')