

# Bayesian Coresets: An Optimization Perspective

The code is used for the experiments in the article, Bayesian Coresets: An Optimization Perspective. We add our IHT 
methods in addition to [Trevor Campbell, etc.'s repository](https://github.com/trevorcampbell/bayesian-coresets) (commit
a7d97b7
on Nov 27, 2019).   
Our method, i.e., A-IHT and A-IHT II are in `bayesiancoresets/coreset/iht_coreset.py`.

### Installation and Dependencies

To install with pip, download the repository and run `pip3 install . --user` in the repository's root folder. Note: this package depends on [NumPy](http://www.numpy.org), [SciPy](https://www.scipy.org), and [SciKit Learn](https://scikit-learn.org).
The examples also depend on [Bokeh](https://bokeh.pydata.org/en/latest) for plotting.

### The Experiments 

The three experiments in our paper are in `bayesiancoresets/examples/riemann_gaussian/` (Synthetic Gaussian posterior inference),
`bayesiancoresets/examples/riemann_linear_regression/` (ayesian Radial Basis Function Regression),
and `bayesiancoresets/examples/riemann_logistic_poisson_regression/` (ayesian logistic and Poisson regression),
respectively.
To run the experiments, simply run the `run.sh` under each directories.

### Citation
TBD