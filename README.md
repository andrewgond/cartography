# Dataset Cartography

Modified code for the paper [Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics](https://aclanthology.org/2020.emnlp-main.746) at EMNLP 2020.

This repository contains implementation of data maps, as well as other data selection baselines, along with notebooks for data map visualizations.

If using, please cite:
```
@inproceedings{swayamdipta2020dataset,
    title={Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics},
    author={Swabha Swayamdipta and Roy Schwartz and Nicholas Lourie and Yizhong Wang and Hannaneh Hajishirzi and Noah A. Smith and Yejin Choi},
    booktitle={Proceedings of EMNLP},
    url={https://arxiv.org/abs/2009.10795},
    year={2020}
}
```
This repository can be used to build Data Maps, like [this one for SNLI using a RoBERTa-Large classifier](./sample/SNLI_RoBERTa.pdf).
![SNLI Data Map with RoBERTa-Large](./sample/SNLI_RoBERTa.png)

### Pre-requisites

This repository is based on the [HuggingFace Transformers](https://github.com/huggingface/transformers) library.
<!-- Hyperparameter tuning is based on [HFTune](https://github.com/allenai/hftune). -->

### Installation

Clone the repository with:

```
git clone https://github.com/okaycoffee/cartography.git
```


Download the Python modules with:

```
 pip install -r requirements.txt
```

Edit batch size and number of epochs to fit computing power in `configs/$TASK.jsonnet`

### Available formatted datasets:

GLUE datasets should go in the relative path `datasets/glue/$TASK`. 

[MNLI Dataset](https://dl.fbaipublicfiles.com/glue/data/MNLI.zip) (Plotting and filtering)

[SNLI Dataset](https://dl.fbaipublicfiles.com/glue/data/SNLI.zip) (Plotting and filtering)

[QNLI Dataset](https://dl.fbaipublicfiles.com/glue/data/QNLI.zip) (Only plotting)

[SST-2 Dataset](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip) (Only plotting)

### Train GLUE-style model and compute training dynamics

To train a GLUE-style model using this repository:

```
python -m cartography.classification.run_glue
    -c configs/$TASK.jsonnet
    --do_train
    --do_eval
    -o $MODEL_OUTPUT_DIR
```
The best configurations for our experiments for each of the `$TASK`s (SNLI, MNLI, QNLI or WINOGRANDE) are provided under [configs](./configs).

This produces a training dynamics directory `$MODEL_OUTPUT_DIR/training_dynamics`, see a sample [here](./sample/training_dynamics/).

*Note:* you can use any other setup to train your model (independent of this repository) as long as you produce the `dynamics_epoch_$X.jsonl` for plotting data maps, and filtering different regions of the data.
The `.jsonl` file must contain the following fields for every training instance:
- `guid` : instance ID matching that in the original data file, for filtering,
- `logits_epoch_$X` : logits for the training instance under epoch `$X`,
- `gold` : index of the gold label, must match the logits array.


### Plot Data Maps

To plot data maps for a trained `$MODEL` (e.g. RoBERTa-Large) on a given `$TASK` (e.g. SNLI, MNLI, QNLI, SST, or WINOGRANDE):

```
python -m cartography.selection.train_dy_filtering
    --plot
    --task_name $TASK
    --model_dir $PATH_TO_MODEL_OUTPUT_DIR_WITH_TRAINING_DYNAMICS
    --model $MODEL_NAME
```

#### Data Map Coordinates

The coordinates for producing RoBERTa-Large data maps for SNLI, QNLI, MNLI and WINOGRANDE, as reported in the paper can be found under `data/data_map_coordinates/`. Each `.jsonl` file contains the following fields for each instance in the train set:
- `guid` : instance ID matching that in the original data file,
- `index`,
- `confidence`,
- `variability`,
- `correctness`.


### Data Selection

To select (different amounts of) data based on various metrics from training dynamics:

```
python -m cartography.selection.train_dy_filtering
    --filter
    --task_name $TASK
    --model_dir $PATH_TO_MODEL_OUTPUT_DIR_WITH_TRAINING_DYNAMICS
    --metric $METRIC
    --data_dir $PATH_TO_GLUE_DIR_WITH_ORIGINAL_DATA_IN_TSV_FORMAT
```

`$METRIC`s include `confidence`, `variability`, `correctness`, `forgetfulness` and `threshold_closeness`; see [paper](https://aclanthology.org/2020.emnlp-main.746) for more details.

To select _hard-to-learn_ instances, set `$METRIC` as "confidence" and for _ambiguous_, set `$METRIC` as "variability". For _easy-to-learn_ instances: set `$METRIC` as "confidence" and use the flag `--worst`.

Will output a directory called "filtered" with different percentages of the examples based on the `$METRIC` in the format of `cartography_$METRIC_$PERCENTAGE`.

### Shuffling Datasets

Shuffle the filtered train.tsv files to remove any order bias with:

```
python scripts/tsv_shuffle.py $INPUT_TSV $OUTPUT_TSV
```
### Using Shuffled/Filtered Datasets

A filtered and/or shuffled train.tsv file can be substituted for the one in `datasets/glue/$TASK/`. New models can be trained on these modified datasets. The train/dev/test accuracies of models trained on various combinations of the SNLI dataset for 6 epochs are shown below: 

|                                                         | Final Training Accuracy (ID) | Dev Accuracy (OOD) | Test Accuracy (OOD) |
| ------------------------------------------------------- | ---------------------------- | ------------------ | ------------------- |
| 100.00%                                                 | 0.8781                       | 0.8948             | 0.8962              |
| 33% random                                              | 0.8565                       | 0.8718             | 0.8705              |
| 33% easy-to-learn                                       | 0.9988                       | 0.8258             | 0.8245              |
| 33% hard-to-learn                                       | 0.523                        | 0.5527             | 0.5333              |
| 33% ambiguous                                           | 0.7231                       | 0.8746             | 0.8804              |
| 17% easy-to-learn<br>17% hard-to-learn                  | 0.7448                       | 0.6469             | 0.5075              |
| 17% easy-to-learn<br>17% ambiguous                      | 0.8779                       | 0.878              | 0.647               |
| 17% hard-to-learn<br>17% ambiguous                      | 0.5792                       | 0.4449             | 0.8771              |
| 11% easy-to-learn<br>11% hard-to-learn<br>11% ambiguous | 0.7286                       | 0.5152             | 0.4367              |

### Contact and Reference

For questions and usage issues, please contact `swabhas@allenai.org`. If you use dataset cartography for research, please cite the [original paper](https://aclanthology.org/2020.emnlp-main.746) as follows:

```
@inproceedings{swayamdipta-etal-2020-dataset,
    title = "Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics",
    author = "Swayamdipta, Swabha  and
      Schwartz, Roy  and
      Lourie, Nicholas  and
      Wang, Yizhong  and
      Hajishirzi, Hannaneh  and
      Smith, Noah A.  and
      Choi, Yejin",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.746",
    doi = "10.18653/v1/2020.emnlp-main.746",
    pages = "9275--9293",
}
```
Copyright [2020] [Swabha Swayamdipta]

