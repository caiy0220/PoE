# Power of Explanations - PoE
Hi there, this repository contains the source code and necessary resources of the paper: \
Power of Explanations: Towards automatic debiasing in hate speech detection \
*Yi Cai, Arthur Zimek, Gerhard Wunder, Eirini Ntoutsi*

## Dependencies
Most of the experiments were carried out on Colab. 
The dependencies are therefore derived from the Colab's environment at the time the code was runned.
- Python: 3.7.13
- Pytorch: 1.12.1+cu113
- boto3: 1.24.68
- pyyaml: 6.0
- tqdm: 4.64.0

## Dataset
The full Gab Hate Corpus can be downloaded via [GHC](https://osf.io/edua3/) under the `./Data` folder,
with the *ghc_train.tsv* containing the training set and *ghc_test.tsv* containing the test & validation set.
A random split is performed on *ghc_test.tsv* for the preparation of the test & validation sets.

Files should be converted after the split into *.jsonl* format, namely ***train.jsonl, dev.jsonl, test.jsonl***,
and stored under the folder: `./data/majority_gab_dataset_25k/` 
(or any other path with the corresponding configuration updated).

## Training
To train the model with *MiD*, simply run the following command, the other two supported modes are `vanilla` and `soc`,
referring to the *vanilla BERT* and the [*baseline*](https://github.com/BrendanKennedy/contextualizing-hate-speech-models-with-explanations).
```commandline
python launch.py --mode mid
```
Configurations on the whole framework can be found under the `./utils` folder with the ending *.yaml*.
Alter the parameters could also be done in the command line instead of editing the configuration file,
e.g. change the random seed and the training mode:
```commandline
python launch.py --seed 1 --mode vanilla
```
the same holds for other parameters in the 3 configuration files.

Detailed information during training will be stored as a list under the root of the workspace in a *.pkl* file,
the list contains: 
1. *loss*
2. *regularized attribution penalty*
3. *debiasing lists over iterations*
4. *attribution and fpp of the corresponding token found by MiD over iterations in class AttrRecord*
5. *attribution and fpp of the corresponding token listed in baseline over iterations in class AttrRecord*

