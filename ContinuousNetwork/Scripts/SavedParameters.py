def hyperparams(percentage_train):
    params = {}
    if percentage_train == 0.9:
        params = {'normalize': True,
                 'networkarch': 0,
                 'act': 'tanh',
                 'n_layers': 5,
                 'hidden_nodes': 190,
                 'lr': 0.0030252919955671568}
    elif percentage_train == 0.4:
        params = {'normalize': True, 
                  'networkarch': 0, 
                  'act': 'sin', 
                  'n_layers': 6, 
                  'hidden_nodes': 169, 
                  'lr': 0.004200891908289064}
    elif percentage_train == 0.2:
        params = {'normalize': True, 
                  'networkarch': 0, 
                  'act': 'tanh', 
                  'n_layers': 6, 
                  'hidden_nodes': 121, 
                  'lr': 0.0049131348704520815}
    else:
        params = {'normalize': True, 
                  'networkarch': 0, 
                  'act': 'tanh', 
                  'n_layers': 5, 
                  'hidden_nodes': 193, 
                  'lr': 0.00454864923403576}
    return params
    