# Neural Networks for the approximation of Euler's elastica

Repository for the paper "Neural Networks for the approximation of Euler's elastica". The preprint of the paper can be found at: https://arxiv.org/abs/2312.00644.

The codebase depends on the dependencies collected in the files "requirements.txt" found in each main folder. To install the necessary packages for each network, run the respective scripts
> pip install -r requirements.txt 

The codes are organized into 3 main folders:
1. **ContinuousNetwork**, collecting experiments from Section 5.1 of the manuscript, 
2. **ContinuousNetworkTheta**, collecting experiments from Section 5.2 of the manuscript, 
2. **DiscreteNetwork**, collecting experiments from Section 4.1 of the manuscript.

We add to each of these subdirectories a short readme file to facilitate running the code.

The folder **DataSets** contains the beam code used to generate the data sets, and the two data sets (*both-ends* and *right-end*) mentioned in the paper.

Citation key:
@article{celledoni2023neural,
      title={Neural networks for the approximation of Euler's elastica}, 
      author={Elena Celledoni and Ergys Çokaj and Andrea Leone and Sigrid Leyendecker and Davide Murari and Brynjulf Owren and Rodrigo T. Sato Martín de Almagro and Martina Stavole},
      journal={arXiv preprint arXiv:2312.00644},
      year={2023}
}
