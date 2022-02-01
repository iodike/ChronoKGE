# Setup

This setup documentation serves as an introduction for extending and
maintaining the T-LowFER framework.

## Project structure

### Folder structure

The `t_lowfer` module consists of the following folders:

- `experiment`: The experiment classes.
  - `run`: Experiments with a single run.
  - `tune`: Experiments with a tuning run.
- `knowledge`: Modules related to data handling and preprocessing.
  - `augmentation`: Augmentation modules.
  - `kg`: Knowledge graph modules.
  - `time`: Time processing modules.
- `model`: The model classes.
  - `model`: Complete network models.
  - `module`: Model modules and extensions.
- `trainer`: The trainer classes.
- `tuner`: The tuning classes.
- `utils`: Additional helper functions 
  - `vars`: Constants and enumerations.
  - `web`: Web utilities.
- `visualization`: Visualization tools.

## Extending T-LowFER

### Adding a new model

1. Add a new model into `model/model` by:
   1. creating a new model which inherits PyTorch's base `nn.Module`
   2. extending an existing model, e.g., inheriting from `BaseModel`
2. Add a new run-experiment to `experiment/run` by:
   1. creating a new experiment
   2. extending an existing model, e.g., inheriting from `BaseExperiment`
      1. Note: all abstract/empty functions must be implemented
3. (**Optional**) Add a new tune-experiment to `experiment/tune`
4. Register the new model and experiment within the `model/models.py` module
   1. Add an entry into the `COLLECTIONS` dictionary, where:
      1. `key`: a unique string identifier used to call the model
      2. `value`: a tuple containing the class names of 
         - model
         - run-experiment
         - tune-experiment

<br>

### Adding a new dataset

1. Save the new dataset under one of the following folders:
   1. `static`: Datasets with static facts (s,p,o)
   2. `temporal`: Datasets with temporal facts (s,p,o,t)
   3. `synthetic`: Datasets which are synthetically created
2. Register a new dataset in the `knowledge/knowledge_base.py` module
   1. Define a new `KnowledgeBase` in the `KnowledgeBases` object
      1. Specify the KB properties according to the dataset:
         1. `name`: a unique string identifier (same name as data-folder)
         2. `genre`: the genre of the KB (`static`, `temporal`, `synthetic`)
         3. `start_date`: (optional) The start date of the KB
         4. `num_days`: (optional) The number of days covered by the KB
         5. `gran`: (optional) The time granularity of the KB
   2. Add the new KB enum into the `ALL_KB` list in the `KnowledgeBases` object
   3. Register the new KB in the appropriate `function attributes`:
      1. `has_timestamps`: KB contains time in form of timestamps
      2. `has_indices`: KB contains entity indices in datasets
      3. `has_labels`: KB contains entity labels in datasets

<br>
