# SST-2 Data

**Stanford Sentiment Treebank 2** — binary sentiment classification dataset from the [GLUE benchmark](https://gluebenchmark.com/).

| File | Rows | Description |
|---|---|---|
| `dev.tsv` | 872 | Full development / validation split — used for evaluation |
| `train_sample.tsv` | 2,000 | Random sample of the training split (full train = 67,349 rows) |

**Columns:** `sentence` (str), `label` (int: 1 = positive, 0 = negative)

**Source:** [HuggingFace — stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2)

**License:** The SST dataset is made available for research purposes.  
Original paper: Socher et al. (2013) — *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank*, EMNLP 2013.
