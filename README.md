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
python3 -m chrono_kge -mc <main_config> run <run_config>
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
python3 -m chrono_kge -mc <main_config> tune <tune_config>
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
  Whether or not to save results.<br>
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

## Knowledge Graphs

### Static KG

| Dataset | #entities | #relations  | supported |
| -------- |:--------:|:--------:|:--------:|
| FB15K     |14,951|1,345| &#10003; |
| FB15K-237 |14,951|237| &#10003; |
| WN18      |40,943|18| &#10003; |
| WN18RR    |40,943|11| &#10003; |

### Temporal KG

| Dataset | #entities | #relations  | #timestamps | startdate  | enddate  | time-granularity  | supported |
| -------- |:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| WIKIDATA12K |12,554|24|1,018|1000-01-01|2018-01-01|yearly| &#10003; |
| YAGO11K     |10,623|10|-|-453|2844|daily| &#10008; |
| YAGO15K     |-|-|-|-|-|yearly| &#10008; |

### Event KG

| Dataset | #entities | #relations  | #timestamps | startdate  | enddate  | time-granularity  | supported |
| -------- |:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| ICEWS14   |7,129|230|365|2014-01-01|2014-12-31|daily| &#10003; |
| ICEWS05-15|10,488|251|4,017|2005-01-01|2015-12-31|daily| &#10003; |
| ICEWS18   |23,033|256|7,272|2018-1-1|2018-10-31|hourly| &#10003; |
| GDELT-20 |500|20|366|2015-04-01|2016-03-31|daily| &#10003; |

### Synthetic KG

| Dataset | #entities | #relations  | #timestamps | start-date  | end-date  | time-granularity  | supported |
| -------- |:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| DUMMY   |10|3|30|####-##-01|####-##-30|daily| &#10003; |

<br>

## Benchmark results

Best overall results are **highlighted**. Best own results are **underlined**.<br>
Model parameters are reported in [PARAMS.md](PARAMS.md).<br>

### ICEWS14

| Method            | Hits@1  | Hits@3  | Hits@10  | MRR     |
| ----------------- |:-------:|:-------:|:--------:|:-------:|
| TransE            | 0.154   | 0.430   | 0.646    | 0.326   |
| DistMult          | 0.325   | 0.498   | 0.668    | 0.441   |
| LowFER            | 0.328   | 0.520   | 0.695    | 0.452   |
||
| TTransE           | 0.074   | -   | 0.601    | 0.255   |
| TA-TransE         | 0.095   | -   | 0.625    | 0.275   |
| TA-DistMult       | 0.366   | -   | 0.686    | 0.477   |
| DE-SimplE         | 0.418   | 0.592   | 0.725    | 0.526   |
||
| ATiSE             | 0.436   | 0.629   | 0.750    | 0.550   |
| TeRo              | 0.468   | 0.621   | 0.732    | 0.562   |
||
| RTFE              | 0.503   | 0.646   | 0.758    | 0.592   |
| TIMEPLEX          | 0.515   | -   | 0.771    | 0.604   |
| TNTComplEx        | 0.460   | 0.610   | 0.740    | 0.560   |
| TuckerTNT         | 0.521   | 0.655   | 0.753    | 0.604   |
| TeMP              | 0.478   | `0.681`   | `0.828`    | 0.601   |
| CluSTeR           | 0.338   | -   | 0.712    | 0.460   |
| TeLM              | `0.545`   | 0.673   | 0.774    | `0.625`   |
||
| LowFER-S          | 0.371   | 0.546   | 0.713    | 0.486   |
||
| LowFER-T          | 0.507   | 0.628   | 0.721    | 0.582   |
| LowFER-CT         | 0.518   | 0.647   | 0.743    | 0.597   |
| LowFER-RT         | 0.512   | 0.645   | 0.745    | 0.594   |
||
| LowFER-TNT        | 0.508   | 0.630   | 0.719    | 0.583   |
| LowFER-CTNT       | 0.525   | 0.652   | 0.740    | 0.602   |
| LowFER-RTNT       | 0.521   | 0.653   | 0.749    | 0.602   |

<br>

### ICEWS05-15

| Method            | Hits@1  | Hits@3  | Hits@10  | MRR      |
| ----------------- |:-------:|:-------:|:--------:|:--------:|
| TransE            | 0.090   | -   | 0.663    | 0.294   |
| DistMult          | 0.337   | -   | 0.691    | 0.456   |
| LowFER            | -   | -   | -    | -   |
||
| TTransE           | 0.084   | -   | 0.616    | 0.271   |
| TA-TransE         | 0.096   | -   | 0.668    | 0.229   |
| TA-DistMult       | 0.346   | -   | 0.728    | 0.477   |
| DE-SimplE         | 0.392   | 0.578   | 0.748    | 0.513   |
||
| ATiSE             | 0.378   | 0.606   | 0.794    | 0.519   |
| TeRo              | 0.469   | 0.668   | 0.795    | 0.586   |
||
| RTFE              | 0.553   | 0.706   | 0.811    | 0.645   |
| TIMEPLEX          | 0.545   | -   | 0.818    | 0.639   |
| TNTComplEx        | 0.500   | 0.650   | 0.780    | 0.600   |
| TuckerTNT         | 0.559   | 0.686   | 0.783    | 0.638   |
| TeMP              | 0.566   | `0.782`   | `0.917`    | `0.691`   |
| CluSTeR           | 0.349   | -   | 0.630    | 0.446   |
| TeLM              | `0.590`   | 0.728   | 0.823    | 0.678   |
||
| LowFER-S          | 0.376    | 0.551   | 0.716    | 0.492    |
||
| LowFER-T          | 0.481    | 0.604   | 0.706    | 0.560    |
| LowFER-CT         | 0.475    | 0.652   | 0.789    | 0.585    |
| LowFER-RT         | 0.512    | 0.628   | 0.715    | 0.583    |
||
| LowFER-TNT        | 0.484    | 0.608   | 0.710    | 0.563    |
| LowFER-CTNT       | 0.485    | 0.658   | 0.795    | 0.592    |
| LowFER-RTNT       | 0.497    | 0.636   | 0.751    | 0.585    |

<br>

### Wikidata12k

| Method            | Hits@1  | Hits@3  | Hits@10  | MRR      |
| ----------------- |:-------:|:-------:|:--------:|:--------:|
| TransE            | 0.100   | 0.238   | 0.339    | 0.222   |
| DistMult          | 0.119   | 0.238   | 0.460    | 0.222   |
| LowFER            | 0.   | 0.   | 0.    | 0.   |
||
| TTransE           | 0.096   | 0.184   | 0.329    | 0.172   |
| TA-TransE         | 0.329   | -   | `0.807`    | 0.484   |
| TA-DistMult       | 0.652   | -   | 0.785    | `0.700`   |
| DE-SimplE         | -   | -   | -    | -   |
||
| ATiSE             | 0.175   | 0.318   | 0.481    | 0.280   |
| TeRo              | 0.198   | 0.329   | 0.507    | 0.299   |
||
| RTFE              | `0.366`   | `0.508`   | 0.619    | 0.454   |
| TIMEPLEX          | 0.227   | -   | 0.532    | 0.333   |
| TNTComplEx        | -   | -   | -    | -   |
| TuckerTNT         | -   | -   | -    | -   |
| TeMP              | -   | -   | -    | -   |
| CluSTeR           | -   | -   | -    | -   |
| TeLM              | 0.231   | 0.360   | 0.542    | 0.332    |
||
| LowFER-S          | 0.143   | 0.254   | 0.452    | 0.239    |
||
| LowFER-T          | 0.245   | 0.351   | 0.491    | 0.324    |
| LowFER-CT         | 0.229   | 0.341   | 0.484    | 0.311    |
| LowFER-RT         | 0.246   | 0.346   | 0.481    | 0.323    |
||
| LowFER-TNT        | 0.244   | 0.352   | 0.501    | 0.325    |
| LowFER-CTNT       | 0.230   | 0.342   | 0.483    | 0.312    |
| LowFER-RTNT       | 0.242   | 0.344   | 0.484    | 0.320    |

<br>

### GDELT-20

| Method            | Hits@1  | Hits@3  | Hits@10  | MRR      |
| ----------------- |:-------:|:-------:|:-------:|:---------:|
| TransE            | -   | -   | -    | -   |
| DistMult          | -   | -   | -    | -   |
| LowFER            | -   | -   | -    | -   |
||
| TTransE           | -   | -   | -    | -   |
| TA-TransE         | -   | -   | -    | -   |
| TA-DistMult       | -   | -   | -    | -   |
| DE-SimplE         | 0.141   | 0.248   | 0.403    | 0.230   |
||
| ATiSE             | -   | -   | -    | -   |
| TeRo              | -   | -   | -    | -   |
||
| RTFE              | 0.212   | 0.319   | 0.464    | 0.297   |
| TIMEPLEX          | -   | -   | -    | -   |
| TNTComplEx        | -   | -   | -    | -   |
| TuckerTNT         | `0.283`   | `0.418`   | `0.576`    | `0.381`   |
| TeMP              | 0.191   | 0.297   | 0.437    | 0.275   |
| CluSTeR           | 0.116   | -   | 0.319    | 0.183   |
| TeLM              | -   | -   | -    | -   |
||
| LowFER-S          | 0.135   | 0.228   | 0.365    | 0.214    |
||
| LowFER-T          | 0.190   | 0.294   | 0.436    | 0.274    |
| LowFER-CT         | 0.202   | 0.311   | 0.459    | 0.288    |
| LowFER-RT         | 0.187   | 0.291   | 0.433   | 0.271     |
||
| LowFER-TNT        | 0.190   | 0.293   | 0.435    | 0.273    |
| LowFER-CTNT       | 0.205   | 0.316   | 0.463    | 0.293    |
| LowFER-RTNT       | 0.183   | 0.286   | 0.428    | 0.266    |

<br>

### Additional benchmarks
For an exhaustive summary of related benchmark results, visit [TKGC Benchmark Results](https://docs.google.com/spreadsheets/d/10vBm4ZNhSjXfeUwrzsLfSv2VtY52FvXn/edit?usp=sharing&ouid=113122486539420123457&rtpof=true&sd=true).

<br>
