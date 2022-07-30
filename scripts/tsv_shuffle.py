import pandas as pd
import sys

if len(sys.argv) != 3:
    print("Usage: tsv_shuffle INPUT_TSV OUTPUT_TSV")
    exit()

print("Importing...")
full_train = pd.read_csv(sys.argv[1], sep='\t', header=0, keep_default_na=False)
print("Imported! Now shuffling...")
shuffled = full_train.sample(frac=1)
print("Shuffled! Now saving...")
shuffled.to_csv(sys.argv[2], sep="\t", index=False)
print("Saved!")
