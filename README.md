

# Bayesian Coresets: An Optimization Perspective

The code is used for the experiments in the manuscript, [Bayesian Coresets: An Optimization Perspective](https://arxiv.org/abs/2007.00715). We add our IHT 
methods in addition to [Trevor Campbell, etc.'s repository](https://github.com/trevorcampbell/bayesian-coresets) (commit
a7d97b7
on Nov 27, 2019).   

### Accelerated IHT for Coreset
Please refer to the manuscript for details regarding the two algorithms, i.e., A-IHT I (Algorithm 1) and A-IHT II (Algorithm 2).

The two methods are available in `IHT_toolbox/accelerated_iht.py` and can be applied directly. Both numpy version and pytorch version are provided. For large-slcae problems, use the pytorch version on GPU for accelaration. 


### Installation and Dependencies

To install the experiment with pip, download the repository and run `pip3 install . --user` in the repository's root folder. Note: this package depends on [NumPy](http://www.numpy.org), [SciPy](https://www.scipy.org), and [SciKit Learn](https://scikit-learn.org).
The examples also depend on [Bokeh](https://bokeh.pydata.org/en/latest) for plotting.

### The Experiments 

The three experiments in our manuscript are in `bayesiancoresets/examples/riemann_gaussian/` (Synthetic Gaussian posterior inference),
`bayesiancoresets/examples/riemann_linear_regression/` (Bayesian Radial Basis Function Regression),
and `bayesiancoresets/examples/riemann_logistic_poisson_regression/` (Bayesian logistic and Poisson regression),
respectively.
To run the experiments, simply run the `run.sh` under each directories.