from setuptools import setup, find_packages
from codecs import open
from os import path
import subprocess
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

here = path.abspath(path.dirname(__file__))
install_reqs = parse_requirements(here + '/requirements.txt', session=False)

try:
    reqs = [str(ir.requirement) for ir in install_reqs]
except:
    reqs = [str(ir.req) for ir in install_reqs]

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

subprocess.run(["bash", "install_torch_nightly.sh"])

setup(
    name='ControlledFollMerDrift',
    version='1.0',
    description='Controlled Schrodinger-Follmer Drift For Approximate Bayesian Inference',
    long_description=long_description,
    url='git@github.com:franciscovargas/ControlledFollMerDrift.git',
    author='Francisco Vargas',
    author_email='vargfran@gmail.com',
    license='MIT',
    keywords='ML, SDE, Bridge, Stochastic Control',
    install_requires=reqs,
)
