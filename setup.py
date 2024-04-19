from setuptools import find_packages, setup

setup(
    name='learn_regress',
    packages=find_packages(include=['learn_regress']),
    version='0.1.0',
    description='My first Python library with Liner Regression algorithms from DeepLearning.AI',
    author='vik.kur@pm.me',
    install_requires=['numpy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)