from setuptools import setup, find_packages

setup(
    name='bayesiancoresets',
    version='1.0',
    description="Coresets for approximate Bayesian inference",
    author='Jacky Y. Zhang, Trevor Campbell',
    author_email='yiboz@illinois.edu, trevor@stat.ubc.ca',
    url='https://github.com/jackyzyb/bayesian-coresets-optimization',
    # packages=['bayesiancoresets'],
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    keywords=['Bayesian', 'inference', 'coreset', 'sparse', 'variational inference', 'Riemann', 'Hilbert',
              'Frank-Wolfe', 'greedy', 'geodesic', 'IHT'],
    platforms='ALL',
)
