# Pytorch Neural Symbolic Machines

This repository is a Pytorch implementation of Google's [neural symbolic 
machines](https://github.com/crazydonkey200/neural-symbolic-machines) 
model (Liang et al., 2018) for weakly-supervised semantic parsing.
Specifically, we implement the [memory augmented policy optimization](https://arxiv.org/abs/1807.02322) 
(MAPO) algorithm for reinforcement learning of semantic parsers on the 
[WikiTableQuestions](https://nlp.stanford.edu/blog/wikitablequestions-a-complex-real-world-question-understanding-dataset/)
environment. This codebase is used in [Yin et al. (2020)](xxx).

**Key Difference with the Official TensorFlow Implementation** This implementation 
differs from the official tensorflow implementation in the following ways:
 
* The original model uses GloVe word embeddings and bi-directional LSTM networks as the encoder of 
utterances and table schemas (table columns). This implementation uses **pre-trained 
BERT/[TaBERT models](http://fburl.com/TaBERT)** using the table representation strategy 
proposed in Yin et al. (2020). It significantly outperforms the LSTM-based model, 
registering **~49.4%** denotation-level exact-match accuracies on the test set (the 
original accuracy is 43.8).
* We revised the [data pre-processing script](https://github.com/crazydonkey200/neural-symbolic-machines/blob/master/table/wtq/preprocess.py), and 
re-processed the WikiTableQuestions dataset. We also turned off the 
`--anonymize_datetime_and_number_entities` option when preprocessing the dataset,
which gave better performance.
* Besides performance improvement, we also optimized the original codebase for general
efficiency improvement. The training speed is ~2 times faster than the original tensorflow implemenation
when using an LSTM encoder.

## Setup

**Install TaBERT** The BERT representation learning layer depends on the [TaBERT model](http://fburl.com/TaBERT), which is a pre-trained language
model for learning representations of natural language utterances and semi-structured Web tables.
First, install TaBERT following the instructions at [here](http://fburl.com/TaBERT).

**Conda Environment** Next, update the `tabert` conda environment shipped with `TaBERT` with
dependencies of this project:
```bash
conda env update --name tabert --file data/env.yml
``` 

## Usage

The following command demos training a MAPO model using BERT (`bert-base-uncased`) as the 
representation learning layer:
```bash
conda activate tabert

OMP_NUM_THREADS=1 python -m \
  table.experiments \
  train \
  seed 0 \
  --cuda \
  --work-dir=runs/demo_run \
  --config=data/config/config.vanilla_bert.json
```

Training BERT requires at least two GPUs, one for the `learner` to perform
asynchronous gradient updates, and another one for `actor`s to perform 
sampling. Since each `actor` maintains its own copy of BERT, running this 
model requires large GPU memory. In our experiments we used 8x 16GB V100 GPUs.

To improve GPU memory efficiency, the model also includes a
`BERT server` mode, which allows all actors to share a single BERT model hosted 
on GPU, while the remaining parameters of `actor`s are hosted on CPU. This 
significantly reduces memory usage, and the model could be trained on two 1080 GPUs
with 11GB memory each. However, since gradient updates of all `actor`s and the
 centralized BERT model are performed asynchronously, this might result in marginal 
 performance drop. To enable the BERT server mode, simply change the 
 `actor_use_table_bert_proxy` property to `true` in the config file:

```bash
OMP_NUM_THREADS=1 python -m \
  table.experiments \
  train \
  seed 0 \
  --cuda \
  --work-dir=runs/demo_run \
  --config=data/config/config.vanilla_bert.json \
  --extra-config='{"actor_use_table_bert_proxy": true}'
```
 
To perform evaluation with a trained model:
```bash
python -m table.experiments.py \
  test \
  --model runs/demo_run/model.best.bin 
  --test-file data/wikitable_questions/wtq_preprocess_0805_no_anonymize_ent/test_split.json \
  --cuda
```
### Training with pre-trained TaBERT models
TBD.