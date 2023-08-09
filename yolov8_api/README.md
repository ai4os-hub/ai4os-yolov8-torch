# PROJECT GUIDELINES, HOW TO STRUCTURE YOUR MODEL SOURCE CODE

DEEPaaS is a tool designed to replace your scripts and notebooks with a
production-ready API. By configuring the `api` folder, you can define the
endpoints and arguments for your model API and CLI interface
(e.g. `predict` and `train`) saving you the time to define custom scripts
for generic operations.

In a MLOps environment without DEEPaaS, you would need to define a script for
each operation you want to perform on your model. For example, you would need
a script to train your model and a script to predict with your model. Those
scripts would need for arguments, for example the model path or how many
epochs to train for. WIth DEEPaaS, you can define those arguments in the API
as Schemas and then use the `deepaas-cli` or `deepaas-server` to execute the
operations.

The straightforward way to use DEEPaaS is to open the `__init__.py` file in
the root of the package and complete the `TODO` sections.

Once you have your functions ready, you can customize your model API and CLI
arguments and responses by editing the `api.schemas` and`api.responses`
modules.

## If you need more API/CLI methods than the ones provided by DEEPaaS

Although DEEPaaS provides a way to define generic operations, you might still
need to define custom scripts for your model. For example, you might want to
create a script to generate datasets from raw data, or a script to compare
different models. In this case, please follow the following guidelines:

- Use the `config.py` module to define CONSTANTS across the AI-project.
- Use an args library (e.g. `argsparse`) to define command line arguments.
- Use the `logging` module to log information about the script's progress.
- DO NOT define generic functions or public methods inside script.

> Generic functions can be defined, for example in `models.__init__` or inside
> a new module called `utils.py`. Generally you want to call scripts
> `actions-like` and modules `things-like`. For example, a correct name for a
> script would be `create_model.py` and a correct name for a module would be
> `utils.py`.

One simple example is to add your scripts in the root of the package:

```
├── make_dataset.py     <- Script to generate processed datasets from raw data
├── make_model.py       <- Script to create models
├── compare_models.py   <- Script to compare models
├── config.py           <- Module for CONSTANTS shared between files
└── __init__.py         <- Package import entrypoint
```

Then you can simple execute the scripts from the command line:

```bash
$ python -m yolov8_api.make_dataset {your_arguments}
$ python -m yolov8_api.make_model {your_arguments}
$ python -m yolov8_api.compare_models {your_arguments}
```

With this simple structure, the user expects to be able to execute the
scripts on from the command line and import the functions defined in the
`__init__.py` module.

> As user, it is not intuitive to import a function from a script such for
> example `from my_project.make_dataset import DataTool`. It is more intuitive
> to simply import `from my_project import DataTool`. This is why we define
> functions in modules and not in scripts. Once the functions are defined,
> users and scripts can both make use of them.

A more complex example, can classify scripts by tasks using different folders.
For example:

```
├── dataset             <- Folder containing multiple dataset scripts
│   └── make_dataset.py     <- Script to generate datasets from raw data
├── models              <- Folder containing multiple model scripts
│   └── make_model.py     <- Script to create models
├── visualization       <- Folder containing multiple visualization scripts
│   └── compare_models.py   <- Script to compare models
├── config.py           <- Module for CONSTANTS shared between files
└── __init__.py         <- Package import entrypoint
```

Then you can execute the scripts from the command line including the folder:

```bash
$ python -m yolov8_api.dataset.make_dataset {your_arguments}
$ python -m yolov8_api.models..make_model {your_arguments}
$ python -m yolov8_api.visualization.compare_models {your_arguments}
```
