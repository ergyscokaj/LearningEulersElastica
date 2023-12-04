This directory contains the codebase for the discrete network. The code is run in Google Colab, with V100 GPU.

The collected files and folders are:
- *main.ipynb* that can be run as it is, and contains the lines of code for defining the model, hyperparameters testing, training the network, and evaluating the model.
- *Scripts/Training.py* is the script implementing the training routine.
- *Scripts/GetData.py* is the script organizing the data set.
- *Scripts/SavedParameters.py* is the script where the combinations of hyperparameters yelding the best results for each experiment are saved.
- *Scripts/PlotResults.py* is the script where the accuracies are measured and the plotting is implemented. 
- *SavedResults.csv* is the file where the different hyperparameter combinations found by Optuna are saved.
- *TrainedModels/BothEnds0._data.pt* are the files containing the trained models for the case of different splittings of the data set both-ends.
- *TrainedModels/BothEndsRightEnd0.9data.pt* is the file containing the trained model for the case where the data set consists of the both-ends+right-end.
- *TrainedModels/BothEndsExtrapolation0.9data.pt* is the file containing the trained model for the extrapolation experiment.

In *main.ipynb*, for the hyperparameters' choice, before training one can choose to:
- do hyperparameter search with Optuna, 
- manually input parameters by setting *manual_input = True* (default is False), 
- or use the combination saved in *Scripts/SavedParameters.py*,
or can skip the training phase by loading the pretraind models from *TrainedModels* and just visualize the results.
