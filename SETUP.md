# Setup

This setup documentation serves as an introduction for extending and
maintaining the ChronoKGE framework.

## Project structure

### Folder structure

The `chrono_kge` module consists of the following folders:

- `main`: The main modules.
  - `handler`: Handler for framework state.
  - `manager`: Managers for framework execution.
  - `parser`: Parsers for framework parameter.
- `experiment`: The experiment modules.
- `knowledge`: Modules related to data handling and preprocessing.
  - `augmentation`: Augmentation modules.
  - `chrono`: Time processing modules.
  - `graph`: Knowledge graph modules.
- `model`: The model modules.
  - `kge`: Complete network models.
    - `deepml`: KGE models using deep learning.
    - `geometric`: KGE-models using distance scoring.
    - `semantic`: KGE-models using similarity scoring.
  - `module`: Model modules and extensions.
    - `calculus`: Basic tensor modules.
    - `embedding`: Embedding modules.
    - `pooling`: Bilinear pooling modules.
    - `regularizer`: Regularization modules.
    - `scoring`: Scoring modules.
- `statistics`: The statistics modules.
- `trainer`: The trainer modules.
- `tuner`: The tuner modules.
- `utils`: Additional helper functions 
  - `vars`: Constants and enumerations.
  - `web`: Web utilities.
- `visualization`: Visualization tools.

### KGE Families

- Deep Learning (network): `model.kge.deepml`
- Geometric (distance): `model.kge.geometric`
- Semantic matching (similarity): `model.kge.semantic`

## Extending ChronoKGE

### Adding a new model

In the following, we describe how to add a new embedding model.

1. Add a new model into `model/kge/<family>`([Learn more](#kge-families)) by either:
   1. creating a new model which inherits PyTorch's base `nn.Module`
   2. extending an existing model, e.g., inheriting from `kge_model.KGE_Model`
2. Register your model within the `model/__model__.py` module
   1. Add an entry into the `REGISTER` dictionary, where:
      1. `key`: a unique string identifier used to call your model
      2. `value`: the class type of your model
3. Done!

<br>

### Adding a new dataset

In the following, we describe how to add a new benchmark dataset.

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
3. Done!

<br>
