def hyperparams(datacase, percentage_train):
    params = {}
    if datacase == 1:
        if percentage_train == 0.9:
            params = {'is_res': False,
                      'normalize': True,
                      'act': 'tanh',
                      'n_layers': 4,
                      'hidden_nodes': 879,
                      'lr': 0.0013780948769418003,
                      'weight_decay': 1.535048308234182e-07,
                      'gamma': 0.004242492350411101,
                      'batch_size': 32}
        elif percentage_train == 0.4:
            params = {'is_res': False, 
                      'normalize': False, 
                      'act': 'tanh', 
                      'n_layers': 3, 
                      'hidden_nodes': 904, 
                      'lr': 0.0014297239550464156, 
                      'weight_decay': 1.3178777419742786e-07, 
                      'gamma': 0.009338834655473796, 
                      'batch_size': 32} 
        elif percentage_train == 0.2:
            params = {'is_res': False, 
                      'normalize': False, 
                      'act': 'tanh', 
                      'n_layers': 4, 
                      'hidden_nodes': 351, 
                      'lr': 0.005455970920050363, 
                      'weight_decay': 2.142731653129661e-07, 
                      'gamma': 0.0029737219856175584, 
                      'batch_size': 32}
        else:
            params = {'is_res': False,
                      'normalize': False,
                      'act': 'tanh',
                      'n_layers': 4,
                      'hidden_nodes': 950,
                      'lr': 0.0010197311392830968,
                      'weight_decay': 1.79067309430248e-06,
                      'gamma': 0.00859530974114581,
                      'batch_size': 64}     
    if datacase == 2:
        params = {'is_res': False,
                  'normalize': True,
                  'act': 'tanh',
                  'n_layers': 4,
                  'hidden_nodes': 879,
                  'lr': 0.0013780948769418003,
                  'weight_decay': 1.535048308234182e-07,
                  'gamma': 0.004242492350411101,
                  'batch_size': 32}
    if datacase == 3:
        params = {'is_res': False,
                  'normalize': True,
                  'act': 'lrelu',
                  'n_layers': 2,
                  'hidden_nodes': 1006,
                  'lr': 0.00361187379687664,
                  'weight_decay': 1.5159394412320704e-07,
                  'gamma': 0.006388763331691748,
                  'batch_size': 32}
    return params
    