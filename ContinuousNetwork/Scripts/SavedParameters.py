def hyperparams(percentage_train):
    params = {}
    if percentage_train == 0.8:
        params = {'n_layers': 6,
                 'hidden_nodes': 106}
    elif percentage_train == 0.4:
        params = {'n_layers': 8, 
                  'hidden_nodes': 181}
    elif percentage_train == 0.2:
        params = {'n_layers': 7, 
                  'hidden_nodes': 185}
    else:
        params = {'n_layers': 6, 
                  'hidden_nodes': 139}
    return params
    