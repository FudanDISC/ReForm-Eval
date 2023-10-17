from setuptools import setup, find_packages

setup(
    name='ReForm-Eval',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'pycocoevalcap',
        'tqdm',
        'scikit-learn'

    ],
    entry_points={
        'console_scripts': [
            'run_eval = run_eval:main',
            'run_loader_eval = run_loader_eval:main',
            'load_reform_dataset = build:load_reform_dataset'
        ],
    },
)