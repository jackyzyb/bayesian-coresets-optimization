
# Bayesian Coresets Construction with Accelerated Iterative Hard Thresholding

This repository provides python implementation for constructing Bayesian coreset using 
 accelerated iterative hard thresholding (A-IHT). 

Corresponding paper:
 [Bayesian Coresets: Revisiting the Optimization Perspective](https://arxiv.org/abs/2007.00715), 
 Jacky Y. Zhang, Rajiv Khanna, Anastasios Kyrillidis, and Oluwasanmi Koyejo. (**AISTATS 2021**)

Please refer to the paper for details regarding the two algorithms, i.e., A-IHT I (Algorithm 1) and A-IHT II (Algorithm 2),
and the experiment settings.

*Contents* 

1. IHT Toolbox
2. Experiments


## IHT Toolbox
Three implementations of the A-IHT are provided in `IHT_toolbox/accelerated_iht.py`, and can be applied directly.
Specifically,
1.  A-IHT I implemented with numpy
2.  A-IHT II implemented with numpy
3.  A-IHT II implemented with pytorch  
For large-scale problems, use the pytorch version on GPU for acceleration. 


## Experiments

The three experiments in our paper are in `experiments/bayesiancoresets/examples/riemann_gaussian/` 
(Synthetic Gaussian posterior inference),
`experiments/bayesiancoresets/examples/riemann_linear_regression/` (Bayesian Radial Basis Function Regression),
and `experiments/bayesiancoresets/examples/riemann_logistic_poisson_regression/` (Bayesian logistic and Poisson regression),
respectively.
To run the experiments, simply run the `run.sh` under each directories.

### Installation and Dependencies

To install the experiment with pip, download the repository and run `pip3 install . --user` in the `experiments/` folder. 
The experiments depend on [NumPy](http://www.numpy.org), [SciPy](https://www.scipy.org), and [SciKit Learn](https://scikit-learn.org).
The examples also depend on [Bokeh](https://bokeh.pydata.org/en/latest) and [cairosvg](https://cairosvg.org/) for plotting.

### Reference
The experiments are built on the framework of [Trevor Campbell, etc.'s repository](https://github.com/trevorcampbell/bayesian-coresets) 
(commit a7d97b7 on Nov 27, 2019).   