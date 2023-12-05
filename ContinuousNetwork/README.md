This directory contains the codebase for the continuous network. The code is run in Google Colab, with V100 GPU and it depends on the dependencies collected in the file "requirements.txt". To install the necessary packages run the script
> pip install -r requirements.txt

The collected files and folders are:
- *main.ipynb* that can be run as it is, and contains the lines of code for hyperparameters testing, training the network, and evaluating the model.
- *Scripts/Network.py* is the script defining the model.
- *Scripts/Training.py* is the script implementing the training routine.
- *Scripts/GetData.py* is the script organizing the data set.
- *Scripts/SavedParameters.py* is the script where the combinations of hyperparameters yelding the best results for each experiment are saved.
- *Scripts/PlotResults.py* is the script where the accuracies are measured and the plotting is implemented. 
- *Scripts/Utils.py* is the script containing a function used in the hyperparameters testing phase.
- *SavedResults.csv* is the file where the different hyperparameter combinations found by Optuna are saved.
- *TrainedModels/BothEnds0._data.pt* are the files containing the trained models for the case of different splittings of the data set *both-ends*.

In *main.ipynb*, for the hyperparameters' choice, before training one can choose to:
- do hyperparameter search with Optuna, 
- manually input parameters by setting *manual_input = True* (default is False), 
- or use the combination saved in *Scripts/SavedParameters.py*,
or can skip the training phase by loading the pretrained models from *TrainedModels* and just visualize the results.