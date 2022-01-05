# PasswordFactory

This program aims to evaluate efficiency of several recurrent neural networks and train algorithms for password generation on coverage and cross entropy loss.

## Setup

To install required libraries, please use following command:

```
pip3 install -r requirements.txt
```

Training and evaluation datasets must be added in `data/TrainEval` directory.
Training file name must be `train.txt`.
Evaluation file name must be `eval.txt`.

## Run program

To train a model, please use following command:

```
python3 password_generation.py -t True
```

To evaluate a model, please use following command:

```
python3 password_generation.py -t False -e True -m <model name>
```

To evaluate coverage, please use `-e True` argument or output files in `runs` directory.


## Datasets

The english words dataset (`words.txt`) which is used in this repository is an open source dataset hosted [here](https://github.com/dwyl/english-words/).
