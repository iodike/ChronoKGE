# ChronoKGE

ChronoKGE is a knowledge graph embedding framework to ease time-focused research in representation learning.

![alt text](https://github.com/iodike/ChronoKGE/blob/master/media/chrono-kge.png?raw=true)

<br>

## Requirements

> `Python 3.6+`
> https://www.python.org/downloads/
> 
> `PyTorch 1.8+`
> https://pytorch.org/get-started/locally/
>
> `Optuna 2.9+`
> https://optuna.readthedocs.io/en/latest/installation.html
>
> `Selenium 3.14+`
> https://selenium-python.readthedocs.io/installation.html

<br>

## Installation

#### Install dependencies
```
pip3 install -r requirements.txt
```

<br>

## Usage

#### Run

> `run` Executes the experiment with the given parameters.

##### Via Command Line:
```
python3 -m chrono_kge [<model-args>] run [<run-args>]
```
For more details, see [model arguments](#1-model-arguments) and [run arguments](#2-run-arguments).

##### Via YAML Config:
```
python3 -m chrono_kge -mc <main_config> run -rc <run_config>
```

<br>

#### Tune

> `tune` Performs parameter tuning with the provided amount of trials.

##### Via Command Line:
```
python3 -m chrono_kge [<model-args>] tune [<tune-args>]
```
For more details, see [model arguments](#1-model-arguments) and [tune arguments](#3-tune-arguments).

##### Via YAML Config:
```
python3 -m chrono_kge -mc <main_config> tune -tc <tune_config>
```

<br>

## Examples

### Example 1

Run with default parameters `model=lowfer-tnt`, `kg=icews14`, `dim=300`, `lr=0.01` using YAML.

```
python3 -m chrono_kge -mc "config/main/default.yaml" run -rc "config/run/default.yaml"
```

### Example 2

Run with default parameters `model=lowfer-tnt`, `kg=icews14`, `dim=300`, `lr=0.01` using CMD.

```
python3 -m chrono_kge -m "t-lowfer" -d "icews14" -am 1 -mm 1 run -lr 0.01 -ed 300
```


<br>

## Optional arguments

### 1. Model arguments

  `-m MODEL, --model MODEL`<br>
  Learning model.<br>
  Supported models: `lowfer`, `tlowfer`.<br>
  Default `tlowfer`.  

  `-d DATASET, --dataset DATASET`<br>
  Which dataset to use.<br>
  Supported datasets: see [knowledge graphs](#knowledge-graphs) below.<br>
  Default `icews14`.

  `-e EPOCHS, --epochs EPOCHS`<br>
  Number of total epochs.<br>
  Default `1000`.

  `-am AUGMENT_MODE, --aug_mode AUGMENT_MODE`<br>
  The mode of augmentation.<br>
  Supported methods: see [augmentation modes](#augmentation-modes) below.<br>
  Default `0`.

  `-rm REG_MODE, --reg_mode REG_MODE`<br>
  The mode of regularization.<br>
  Supported methods: see [regularization modes](#regularization-modes) below.<br>
  Default `0`.

  `-mm MODULATION_MODE, --mod_mode MODULATION_MODE`<br>
  Modulation mode.<br>
  Supported modulations: see [modulation modes](#modulation-modes) below.<br>
  Default `0`.

  `-em ENC_MODE, --enc_mode ENC_MODE`<br>
  Supported methods: see [encoding modes](#encoding-modes) below.<br>
  Default `0`.

  `-c, --cuda`<br>
  Whether to use cuda (GPU) or not (CPU).<br>
  Default `CPU`.

  `--save`<br>
  Whether to save results.<br>
  Default `False`.

<br>

## Augmentation modes
`0`: `None`<br>
`1`: `Reverse triples`<br>
`2`: `Back translation (pre)`<br>
`3`: `Back translation (ad-hoc)`<br>
`4`: `Reverse triples` + `Back translation (pre)`<br>

<br>

`2`: Augmentation using precomputed translations.

`3`: Ad-hoc back translation using free `Google Translate` service.<br>
High confidence, max. 2 translations, language order `ch-zn`, `es`, `de`, `en`.<br>
Supported KB: `ICEWS14`, `ICEWS05-15`, `ICEWS18`

<br>

## Regularization modes

`0`: `None`<br>
`1`: `Omega` (embedding regularization)<br>
`2`: `Lambda` (time regularisation)<br>
`3`: `Omega` + `Lambda`<br>

`*` Tensor norms: `Omega: p=3`, `Lambda: p=4`

<br>

## Modulation modes

### 1. Time Modulation (T)
Extends LowFER with `dynamic` relations.<br>
`mode=0`

### 2. Time-no-Time Modulation (TNT)
Extends LowFER with `dynamic` and `static` relations.<br>
`mode=1`

<br>

## Benchmark results

Results for semantic matching models on ICEWS14 and ICEWS05-15.

### ICEWS14

| Method            | MRR     | H@10    | H@3      | H@1     |
| ----------------- |:-------:|:-------:|:--------:|:-------:|
| DE-DistMult       | 0.501   | 0.708   | 0.569    | 0.392   |
| DE-SimplE         | 0.526   | 0.725   | 0.592    | 0.418   |
| TComplEx          | 0.560   | 0.730   | 0.610    | 0.470   |
| TNTComplEx        | 0.560   | 0.740   | 0.610    | 0.460   |
| TuckERT           | 0.594   | 0.731   | 0.640    | 0.518   |
| TuckERTNT         | 0.604   | 0.753   | 0.655    | 0.521   |
||
|LowFER-T           | 0.584   | 0.734   | 0.630    | 0.505   |
|LowFER-TNT         | 0.586   | 0.735   | 0.632    | 0.507   |
|LowFER-CFB         | 0.623   | 0.757   | 0.671    | 0.549   |
|LowFER-FTP         | 0.617   | 0.765   | 0.665    | 0.537   |

<br>

### ICEWS05-15

| Method            | MRR     | H@10    | H@3      | H@1     |
| ----------------- |:-------:|:-------:|:--------:|:-------:|
| DE-DistMult       | 0.484   | 0.718   | 0.546    | 0.366   |
| DE-SimplE         | 0.513   | 0.748   | 0.578    | 0.392   |
| TComplEx          | 0.580   | 0.760   | 0.640    | 0.490   |
| TNTComplEx        | 0.600   | 0.780   | 0.650    | 0.500   |
| TuckERT           | 0.627   | 0.769   | 0.674    | 0.550   |
| TuckERTNT         | 0.638   | 0.783   | 0.686    | 0.559   |
||
|LowFER-T           | 0.559   | 0.714   | 0.605    | 0.476   |
|LowFER-TNT         | 0.562   | 0.717   | 0.608    | 0.480   |
|LowFER-CFB         | 0.638   | 0.791   | 0.690    | 0.555   |
|LowFER-FTP         | 0.625   | 0.792   | 0.681    | 0.534   |

<br>

### Additional benchmarks
For an exhaustive summary of related benchmark results, visit [TKGC Benchmark Results](https://docs.google.com/spreadsheets/d/10vBm4ZNhSjXfeUwrzsLfSv2VtY52FvXn/edit?usp=sharing&ouid=113122486539420123457&rtpof=true&sd=true).

<br>

## Citation

If you find our work useful, please consider citing.

```bibtex
@inproceedings{dikeoulias-etal-2022-tlowfer,
    title = "Temporal Knowledge Graph Reasoning with Low-rank and Model-agnostic Representations",
    author = "Dikeoulias, Ioannis and
    Amin, Saadullah and 
    Neumann, Günter",
    booktitle = "Proceedings of the 7th Workshop on Representation Learning for NLP",
    month = may,
    year = "2022",
    address = "Online",
    publisher = "Association for Computational Linguistics"
}
}
```
