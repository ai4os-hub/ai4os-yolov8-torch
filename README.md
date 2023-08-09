# yolov8_api

[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/yolov8_api/master)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/yolov8_api/job/master)

add api to yolov8

To launch it, first install the package then run [deepaas](https://github.com/indigo-dc/DEEPaaS):

```bash
git clone https://git.scc.kit.edu/m-team/ai/yolov8_api
cd yolov8_api
pip install -e .
deepaas-run --listen-ip 0.0.0.0
```

The associated Docker container for this module can be found in https://git.scc.kit.edu/m-team/ai/DEEP-OC-yolov8_api.

## Project structure

```
├── Jenkinsfile             <- Describes basic Jenkins CI/CD pipeline
├── LICENSE                 <- License file
├── README.md               <- The top-level README for developers using this project.
├── VERSION                 <- Version file indicating the version of the model
│
├── yolov8_api
│   ├── README.md           <- Instructions on how to integrate your model with DEEPaaS.
│   ├── __init__.py         <- Makes <your-model-source> a Python module
│   ├── ...                 <- Other source code files
│   └── config.py           <- Module to define CONSTANTS used across the AI-model python package
│
├── api                     <- API subpackage for the integration with DEEP API
│   ├── __init__.py         <- Makes api a Python module, includes API interface methods
│   ├── config.py           <- API module for loading configuration from environment
│   ├── responses.py        <- API module with parsers for method responses
│   ├── schemas.py          <- API module with definition of method arguments
│   └── utils.py            <- API module with utility functions
│
├── data                    <- Data subpackage for the integration with DEEP API
│   ├── external            <- Data from third party sources.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
│
├── docs                   <- A default Sphinx project; see sphinx-doc.org for details
│
├── models                 <- Folder to store your models
│
├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
│                             the creator's initials (if many user development),
│                             and a short `_` delimited description, e.g.
│                             `1.0-jqp-initial_data_exploration.ipynb`.
│
├── references             <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures            <- Generated graphics and figures to be used in reporting
│
├── requirements-dev.txt    <- Requirements file to install development tools
├── requirements-test.txt   <- Requirements file to install testing tools
├── requirements.txt        <- Requirements file to run the API and models
│
├── pyproject.toml         <- Makes project pip installable (pip install -e .)
│
├── tests                   <- Scripts to perform code testing
│   ├── configurations      <- Folder to store the configuration files for DEEPaaS server
│   ├── conftest.py         <- Pytest configuration file (Not to be modified in principle)
│   ├── data                <- Folder to store the data for testing
│   ├── models              <- Folder to store the models for testing
│   ├── test_deepaas.py     <- Test file for DEEPaaS API server requirements (Start, etc.)
│   ├── test_metadata       <- Tests folder for model metadata requirements
│   ├── test_predictions    <- Tests folder for model predictions requirements
│   └── test_training       <- Tests folder for model training requirements
│
└── tox.ini                <- tox file with settings for running tox; see tox.testrun.org
```

## Integrating your model with DEEPaaS

After executing the cookiecutter template, you will have a folder structure
ready to be integrated with DEEPaaS. The you can decide between starting the
project from scratch or integrating your existing model with DEEPaaS.

The folder `yolov8_api` is designed to contain the source
code of your model. You can add your model files there or replace it by another
repository by using [git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules).
The only requirement is that the folder `yolov8_api` contains
an `__init__.py` file conserving the already defined methods. You can edit the
template functions already defined inside or import your own functions from
another file. See the [README.md](./yolov8_api/README.md)
in the `yolov8_api` folder for more information.

Those methods, are used by the subpackage `api` to define the API interface.
See the project structure section for more information about the `api` folder.
You are allowed to customize your model API and CLI arguments and responses by
editing `api.schemas` and`api.responses` modules. See documentation inside those
files for more information.

## Documentation

TODO: Add instructions on how to build documentation

## Testing

Testing process is automated by tox library. You can check the environments
configured to be tested by running `tox --listenvs`. If you are missing one
of the python environments configured to be tested (e.g. py310, py39) and
you are using `conda` for managing your virtual environments, consider using
`tox-conda` to automatically manage all python installation on your testing
virtual environment.

Tests are implemented following [pytest](https://docs.pytest.org) framework.
Fixtures and parametrization are placed inside `conftest.py` files meanwhile
assertion tests are located on `test_*.py` files. As developer, you can edit
any of the existing files or add new ones as needed. However, the project is
designed so you only have to edit the files inside:

    - tests/data: To add your testing data (small datasets, etc.).
    - tests/models: To add your testing models (small models, etc.).
    - tests/test_metadata: To fix and test your metadata requirements.
    - tests/test_predictions: To fix and test your predictions requirements.
    - tests/test_training: To fix and test your training requirements.

The folder `tests/data` should contain minimalistic but representative
datasets to be used for testing. In a similar way, `tests/models` should
contain simple models for testing that can fit on your code repository. This
is important to avoid large files on your repository and to speed up the
testing process.

Running the tests with tox:

```bash
$ pip install -r requirements-dev.txt
$ tox
```

Running the tests with pytest:

```bash
$ pip install -r requirements-test.txt
$ python -m pytest --numprocesses=auto --dist=loadscope tests
```
