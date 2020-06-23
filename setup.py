from setuptools import setup, find_packages

setup(
    name = 'bayesiancoresets',
    version='0.9.1',
    description="Coresets for approximate Bayesian inference",
    #packages=['bayesiancoresets'],
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    keywords = ['Bayesian', 'inference', 'coreset', 'iht', 'sparse', 'variational inference', 'Riemann',  'Hilbert', 'Frank-Wolfe', 'greedy', 'geodesic'],
    platforms='ALL',
)
