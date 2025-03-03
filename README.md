# Dataset Cartography

Institute for Computing in Research project. The paper is available [here](https://raw.githubusercontent.com/andrewgond/cartography/main/documents/BishnuGondoputroDataMapsPaper.pdf), and the slideshow is available [here](https://raw.githubusercontent.com/andrewgond/cartography/main/documents/BishnuGondoputroDataMapsSlides.pdf).

This repository can be used to build Data Maps, like [this one for SNLI using a RoBERTa-Large classifier](./sample/SNLI_RoBERTa.pdf).
![SNLI Data Map with RoBERTa-Large](./sample/SNLI_RoBERTa.png)

### Pre-requisites

This repository is based on the [HuggingFace Transformers](https://github.com/huggingface/transformers) library.
<!-- Hyperparameter tuning is based on [HFTune](https://github.com/allenai/hftune). -->

### Installation

Clone the repository with:

```
git clone https://github.com/andrewgond/cartography.git
```

Enter the `cartography` directory with:
```
cd cartography
```

Download the Python modules with:

```
 pip install -r requirements.txt
```

Edit the batch size and number of epochs for your system in `configs/$TASK.jsonnet`. You will need at least 4 GiB of VRAM regardless of configuration.

### Available datasets:

GLUE datasets should go in the relative path `datasets/glue/$TASK`. 

[MNLI Dataset](https://dl.fbaipublicfiles.com/glue/data/MNLI.zip) (Plotting and filtering)

[SNLI Dataset](https://dl.fbaipublicfiles.com/glue/data/SNLI.zip) (Plotting and filtering)

[QNLI Dataset](https://dl.fbaipublicfiles.com/glue/data/QNLI.zip) (Plotting only)

[SST-2 Dataset](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip) (Plotting only)

For example, to get the SNLI dataset:
```
mkdir -p datasets/glue/
cd datasets/glue/
wget https://dl.fbaipublicfiles.com/glue/data/SNLI.zip
unzip SNLI.zip
```

### Automated run (SNLI)

For an automated run of our experiment:
```
python -m scripts.automated
```
Results will be diplayed in the terminal once complete.

### Train GLUE-style model and compute training dynamics

To train a GLUE-style model using this repository:

```
python -m cartography.classification.run_glue
    -c configs/$TASK.jsonnet
    --do_train
    --do_eval
    -o $MODEL_OUTPUT_DIR
```
The best configurations for our experiments for each of the `$TASK`s (SNLI, MNLI, QNLI, SST-2, or WINOGRANDE) are provided under [configs](./configs).

This produces a training dynamics directory `$MODEL_OUTPUT_DIR/training_dynamics`, see a sample [here](./sample/training_dynamics/).

*Note:* you can use any other setup to train your model (independent of this repository) as long as you produce the `dynamics_epoch_$X.jsonl` for plotting data maps, and filtering different regions of the data.
The `.jsonl` file must contain the following fields for every training instance:
- `guid` : instance ID matching that in the original data file, for filtering,
- `logits_epoch_$X` : logits for the training instance under epoch `$X`,
- `gold` : index of the gold label, must match the logits array.


### (Optional) Plot Data Maps

To plot data maps for a trained `$MODEL` (e.g. RoBERTa-Large) on a given `$TASK` (e.g. SNLI, MNLI, QNLI, SST-2, or WINOGRANDE:

```
python -m cartography.selection.train_dy_filtering
    --plot
    --task_name $TASK
    --model_dir $PATH_TO_MODEL_OUTPUT_DIR_WITH_TRAINING_DYNAMICS
    --model $MODEL_NAME
```

#### Data Map Coordinates

The coordinates for producing RoBERTa-Large data maps for SNLI, QNLI, MNLI and WINOGRANDE, as reported in the original paper can be found under `data/data_map_coordinates/`. Each `.jsonl` file contains the following fields for each instance in the train set:
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
| 100.00%                                                 | 0.8936                       | 0.8997             | 0.8976              |
| 33.33% random                                              | 0.8782                       | 0.8776             | 0.8785              |
| 33.33% easy-to-learn                                       | 0.9996                       | 0.8293             | 0.8286              |
| 33.33% hard-to-learn                                       | 0.5680                        | 0.5966             | 0.5856              |
| 33.33% ambiguous                                           | 0.7684                       | 0.8878             | 0.8894              |
| 16.67% easy-to-learn<br>16.67% hard-to-learn                  | 0.7581                       | 0.5938             | 0.5999              |
| 16.67% easy-to-learn<br>16.67% ambiguous                      | 0.8835                       | 0.8802              | 0.8807               |
| 16.67% hard-to-learn<br>16.67% ambiguous                      | 0.5900                       | 0.4076             | 0.4023              |
| 11.11% easy-to-learn<br>11.11% hard-to-learn<br>11.11% ambiguous | 0.7401                       | 0.5249             | 0.5145              |

(Original Code)

Copyright [2020] [Swabha Swayamdipta]
