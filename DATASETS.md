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
