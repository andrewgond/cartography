# automated script to reproduce our work

import os
import pandas as pd
import sys

# set up SNLI dataset

os.system("mkdir -p datasets/glue/")
os.chdir("datasets/glue/")
os.system("wget https://dl.fbaipublicfiles.com/glue/data/SNLI.zip")
os.system("unzip SNLI.zip")
os.chdir("../..")

# initial training + filtering

os.system("python -m cartography.classification.run_glue -c configs/snli.jsonnet --do_train --overwrite_output_dir -o demo-model")
os.system("python -m cartography.selection.train_dy_filtering --plot --task_name SNLI --model_dir demo-model --model Normal-DistilRoBERTa") # optional; generates plot in cartography
os.system("python -m cartography.selection.train_dy_filtering --filter --task_name SNLI --model_dir demo-model --metric confidence --worst --data_dir datasets/glue/")
os.system("mv filtered/ easy-to-learn/")
os.system("python -m cartography.selection.train_dy_filtering --filter --task_name SNLI --model_dir demo-model --metric confidence --data_dir datasets/glue/")
os.system("mv filtered/ hard-to-learn/")
os.system("python -m cartography.selection.train_dy_filtering --filter --task_name SNLI --model_dir demo-model --metric variability --data_dir datasets/glue/")
os.system("mv filtered/ ambiguous/")

# create the new train.tsv files

data = pd.read_csv("datasets/glue/SNLI/train.tsv", sep='\t', header=0, keep_default_na=False, usecols=range(11))
shuffled = data.sample(frac=1)
shuffled.to_csv("full.tsv", sep="\t", index=False)

data = pd.read_csv("datasets/glue/SNLI/train.tsv", sep='\t', header=0, keep_default_na=False, usecols=range(11))
shuffled = data.sample(frac=0.3333)
shuffled.to_csv("rand.tsv", sep="\t", index=False)

data = pd.read_csv("easy-to-learn/cartography_confidence_0.33/SNLI/train.tsv", sep='\t', header=0, keep_default_na=False, usecols=range(11))
shuffled = data.sample(frac=1)
shuffled.to_csv("easy.tsv", sep="\t", index=False)

data = pd.read_csv("hard-to-learn/cartography_confidence_0.33/SNLI/train.tsv", sep='\t', header=0, keep_default_na=False, usecols=range(11))
shuffled = data.sample(frac=1)
shuffled.to_csv("hard.tsv", sep="\t", index=False)

data = pd.read_csv("ambiguous/cartography_variability_0.33/SNLI/train.tsv", sep='\t', header=0, keep_default_na=False, usecols=range(11))
shuffled = data.sample(frac=1)
shuffled.to_csv("ambi.tsv", sep="\t", index=False)

data1 = pd.read_csv("easy-to-learn/cartography_confidence_0.17/SNLI/train.tsv", sep='\t', header=0, keep_default_na=False, usecols=range(11))
data2 = pd.read_csv("hard-to-learn/cartography_confidence_0.17/SNLI/train.tsv", sep='\t', header=0, keep_default_na=False, usecols=range(11))
data = pd.concat([data1, data2])
shuffled = data.sample(frac=1)
shuffled.to_csv("easyhard.tsv", sep="\t", index=False)

data1 = pd.read_csv("easy-to-learn/cartography_confidence_0.17/SNLI/train.tsv", sep='\t', header=0, keep_default_na=False, usecols=range(11))
data2 = pd.read_csv("ambiguous/cartography_variability_0.17/SNLI/train.tsv", sep='\t', header=0, keep_default_na=False, usecols=range(11))
data = pd.concat([data1, data2])
shuffled = data.sample(frac=1)
shuffled.to_csv("easyambi.tsv", sep="\t", index=False)

data1 = pd.read_csv("hard-to-learn/cartography_confidence_0.17/SNLI/train.tsv", sep='\t', header=0, keep_default_na=False, usecols=range(11))
data2 = pd.read_csv("ambiguous/cartography_variability_0.17/SNLI/train.tsv", sep='\t', header=0, keep_default_na=False, usecols=range(11))
data = pd.concat([data1, data2])
shuffled = data.sample(frac=1)
shuffled.to_csv("hardambi.tsv", sep="\t", index=False)

data1 = pd.read_csv("easy-to-learn/cartography_confidence_0.11/SNLI/train.tsv", sep='\t', header=0, keep_default_na=False, usecols=range(11))
data2 = pd.read_csv("hard-to-learn/cartography_confidence_0.11/SNLI/train.tsv", sep='\t', header=0, keep_default_na=False, usecols=range(11))
data3 = pd.read_csv("ambiguous/cartography_variability_0.11/SNLI/train.tsv", sep='\t', header=0, keep_default_na=False, usecols=range(11))
data = pd.concat([data1, data2, data3])
shuffled = data.sample(frac=1)
shuffled.to_csv("easyhardambi.tsv", sep="\t", index=False)

# training on new sets

for model_name in ["full", "rand", "easy", "hard", "ambi", "easyhard", "easyambi", "hardambi", "easyhardambi"]: 
    os.system("mv " + model_name + ".tsv datasets/glue/SNLI/train.tsv")
    os.system("python -m cartography.classification.run_glue -c configs/snli.jsonnet --do_train --do_eval --do_test --overwrite_output_dir -o model-" + model_name)
    os.system("rm -rf datasets/glue/SNLI/cache*")

# print final accuracies

for model_name in ["full", "rand", "easy", "hard", "ambi", "easyhard", "easyambi", "hardambi", "easyhardambi"]: 
    print("\n" + model_name)
    train_acc = pd.read_json("model-" + model_name + "/eval_metrics_train.json", lines=True)["train_acc"][3]
    print("train accuracy: " + str(train_acc))
    dev_acc = pd.read_json("model-" + model_name + "/eval_metrics_snli_dev_.json", lines=True)["acc"][0]
    print("dev accuracy: " + str(dev_acc))
    test_acc = pd.read_json("model-" + model_name + "/eval_metrics_snli_test_.json", lines=True)["acc"][0]
    print("test accuracy: " + str(test_acc))