# efficient-statistical
## Requirements
This code was tested on a Ubuntu 20.04 system with Python 3.7.9..
The python essential requirements are TensorFlow (v2), numpy, scipy and pandas.
To run experiments on the ACAS-Xu data set the onnx and onnxruntime libraries are also needed.
All the explicit requirements are listed in the requirements.txt and can be directly installed/check 
using the pip command. 



## Runing the papers experiments 
Implementation and experiments for the paper "Efficient Statistical Assessment of Neural Network Corruption Robustness"
Experiment j of the paper can be run using the associated exp_5_{j}.py script. 
By default, this will run both Last Particle and ERAN with pre-selected parameters used for the paper. It is also possible to only run the Last Particle or ERAN method using the check_eran and check_lp options.

All the results are given as csv files (added to the logs) which can be aggregated, e.g. using pandas, to obtain averege results.

## Running differents experiments
You can also run different experiments with the Last Particle algorithm using directly the scripts lp_acasxu.py, lp_mnist.py and lp_imagenet.py. You can select parameters by passing them as C-style options.
For now the options implemented are:
--N: the number of particles used
--p_c: the critical probability level
--n_repeat: the number of runs for each experiments
--T: the number of time the kernel is applied after each particle regeneration
--alpha: the confidence level of the test

In addition for lp_acasxu.py there is a C-style option 'properties' to check only certain ACAS Xu properties. 
For example, the command "python -m lp_acasxu.py --N=5 --properties=[1,2,3]" will run an experiment only check the 3 first properties with a 5 particles system.

Similarly, for MNIST and ImageNet one can choose the range of epsilon test using the --epsilon_range option.
