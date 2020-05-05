# Training RNN Language Models with Explicit Negative Examples

This repository is an implementation of the paper [An analysis of the utility of explicit negative examples to improve the syntactic abilities of neural language models (Noji and Takamura 2019)](https://arxiv.org/abs/2004.02451).

The repository contains the code for syntactic evaluation as a submodule, which we modify slightly for our experiments.
To download including this model, run the following command.
```
git clone --recursive https://github.com/aistairc/lm_syntax_negative.git
```

Or, if you already downloaded without `--recursive`, you can add the submodule by:
```
git submodule update --init --recursive
```

## Requirements

We recommend to build an enviroment on `virtualenv` or `conda`.
The tested Python version is 3.6. Following python moduels are required to be installed via pip.

- [PyTorch](https://pytorch.org) (tested version is 1.3.1, but later versions will probably work)
- [inflect](https://pypi.org/project/inflect/) (4.1.0)
- [progress](https://pypi.org/project/progress/)

The following modules are required for data preprocessing.
- [stanford-corenlp](https://pypi.org/project/stanford-corenlp/) (3.9.2)
- tqdm

In addition to these, the training code also supports the use of NVIDIA [Apex](https://github.com/NVIDIA/apex). When `apex` is installed, by giving `--amp` to `train_lm.py`, the models are trained with mixed precision, which we found speed up the training by 1.3-1.5 times.

## Data Preparation

For training LMs, we used the same data as [Gulordava et al. (2018)](https://github.com/facebookresearch/colorlessgreenRNNs), which is a collection of sentences sampled from English Wikipedia (80M/10M/10M tokens for train, valid, and test).

Here we instruct how to train LSTMs with negative examples on this dataset but you could train your models on a different dataset.
We assume the input data is a list of sentences, one sentence per line, as opposed to the data used for document-level LMs, such as WikiText-103.

First download the original data on `data` directory.
```
mkdir -p data/gulordava_wiki; cd data/gulordava_wiki
wget https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/train.txt
wget https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/valid.txt
wget https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/test.txt
cd ../../
```

### Preparing negative examples

Finding target verbs for negative examples require running Stanford CoreNLP. You need to download it first. We used the version 3.9.1.
```
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip
unzip stanford-corenlp-full-2018-02-27.zip
```

Then, run the following commands to obtain the files recording negative examples (`negative_*.train.txt.gz`).

```
export CORENLP_HOME=stanford-corenlp-full-2018-02-27
python scripts/gen_negative_agreements.py --source data/gulordava_wiki/train.txt --output data/gulordava_wiki/negative_agreements.train.txt.gz
python scripts/gen_negative_reflexives.py --source data/gulordava_wiki/train.txt --output data/gulordava_wiki/negative_reflexives.train.txt.gz
```

## Training LSTM-LMs

The following command replicates training of the baseline LSTM-LM.
```
mkdir -p models
python lm/train_lm.py --data data/gulordava_wiki --save models/lstm.pt --mode sentence \
    --shuffle --length-bucket --non-average --plateau-lr-decay \
    --gpu 0 --seed 1111
```
Please replace `--gpu 0` with the number of CUDA device you are using. Also, add `--amp`, if `apex` is available (see [Requirements](#requirements)).
We experimented on five different seeds (1111, 111, 11, 12345, 54321) and reported averaged [syntactic scores](#syntactic_evaluation) in our paper.
A single training run takes around 26 hours on a single Tesla V100 GPU (16GB) without `--amp`.

### Training with additional margin losses

The following is the command for training the model with the same architecture but with additonal token-level margin loss (margin value is `10`).
This setting works the best in our experiments in the paper.
The training time is almost the same as the original LSTM-LM.
```
python lm/train_lm.py --data data/gulordava_wiki --save models/token_margin=10.pt --mode sentagree \
    --neg-mode token --target-syntax agreement reflexive \
    --neg-criterion margin --margin 10 \
    --shuffle --length-bucket --non-average --plateau-lr-decay \
    --gpu 0 --seed 1111
```

The following is for training with another loss which we call "Sentence-level margin loss" in the paper.
This is multi-task learning, and it takes longer time for each epoch.
In our case, single training requires 30 hours on the same machine without `--amp`.
Since this loss in inferior in terms of both efficiency and performance, we do not recommend to use it in practice.
```
python lm/train_lm.py --data data/gulordava_wiki --save models/sent_margin=10.pt --mode sentagree \
    --neg-mode mtl --target-syntax agreement reflexive --half-agreement-batch \
    --neg-criterion margin --margin 10 \
    --shuffle --length-bucket --non-average --plateau-lr-decay \
    --gpu 0 --seed 1111
```

### Training with unlikelihood losses

The following is the command for training with a related loss, unlikelihood loss proposed in [Welleck et al. (2020)](https://openreview.net/forum?id=SJeYe0NtvH).
`--agreement-loss-alpha 1000` means we amplify the negative loss by 1,000 times.
Even with this, as we discussed in the paper, this loss does not outperform our proposed margin loss.
```
python lm/train_lm.py --data data/gulordava_wiki --save models/unlikelihood=1000.pt --mode sentagree \
    --neg-mode token --target-syntax agreement reflexive \
    --neg-criterion unlikelihood --agreement-loss-alpha 1000 \
    --shuffle --length-bucket --non-average --plateau-lr-decay \
    --gpu 0 --seed 1111
```

## Syntactic evaluation

We follow the same procedure as the evaluation code in the original [LM_syneval](https://github.com/BeckyMarvin/LM_syneval) by Marvin and Linzen (2018).
Our `LM_syneval` directory includes some modification to externally call the script to run our trained LMs on the test sentences.

Here we show how to evaluate the syntactic performance of the baseline LSTM (`models/lstm.pt`).
Evaluation comprises of two steps:

The first step runs the trained LM on all test sentences and output log probabilities.
```
export LM_OUTPUT=syneval_out/lstm
mkdir -p ${LM_OUTPUT}
python LM_syneval/src/LM_eval.py --model models/lstm.pt --model_type myrnn --template_dir LM_syneval/EMNLP2018/templates --myrnn_dir lm --lm_output ${LM_OUTPUT} --capitalize --gpu 0
```

Then, the second step calculates the accuracies on each syntactic constructions.
```
python LM_syneval/src/analyze_results.py --results_file syneval_out/lstm/results.pickle --model_type rnn --out_dir ${LM_OUTPUT} --mode full
```
The overall scores are stroed on `syneval_out/lstm/rnn/full_sent/overall_accs.txt`.

**NOTE**: each original test sentence is uncased.
We modify this and capitalize the first token before running LMs, assuming that uncased sentences are surprising and unnatural for LMs trained on the cased sentences.
This may provide some advantages and may partially explain some unexpected performance gap between our baseline LSTMs and other models that are also tuned but not performed well, such as [van Schijndel et al.](https://www.aclweb.org/anthology/D19-1592/), as pointed out by a recent [paper](https://arxiv.org/abs/2005.00187).

## Citation

This code in this repo is used in the paper below. Please cite it if you use the code in your paper.

```
@inproceedings{noji-et-al-2020-lm-syntax-negative,
    title = "An analysis of the utility of explicit negative examples to improve the syntactic abilities of neural language models",
    author = "Noji, Hiroshi  and  Takamura, Hiroya",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Seattle, Washington, USA",
    publisher = "Association for Computational Linguistics"
}

```

## Credits

The code in `lm` directory is originally derived from the LM training code for [AWD-LSTM-LM](https://github.com/salesforce/awd-lstm-lm) by Salesforce Research, released under the BSD-3-Clause License.

## License

MIT
