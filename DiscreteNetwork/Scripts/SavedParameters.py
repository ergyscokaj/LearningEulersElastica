import pandas as pd
import numpy as np

def hyperparams(case,pctg):
    
    best_vals = pd.DataFrame()
    best_vals["percentage_train"] = np.array([0.1,0.2,0.4,0.8,0.8])
    best_vals["datacase"] = np.array([1,1,1,1,3])
    best_vals["gamma"] = np.array([0.007044405451814177,0.006335851468590373,0.009004175808977003,0.003853035138801786
    ,0.0073229668983443436
    ])
    best_vals["n_layers"] = np.array([4,4,4,4,3])
    best_vals["hidden_nodes"] = np.array([950,978,997,985,616])
    
    vals = best_vals[(best_vals['datacase'] == case) & (best_vals['percentage_train'] == pctg)]
    
    nlayers = vals.iloc[0]["n_layers"]
    hidden_nodes = vals.iloc[0]["hidden_nodes"]
    gamma = vals.iloc[0]["gamma"]
    
    params = {'n_layers': int(nlayers),
            'hidden_nodes': int(hidden_nodes),
            'gamma': gamma}
    
    return params