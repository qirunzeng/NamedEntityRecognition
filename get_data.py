from datasets import load_dataset

raw_datasets = load_dataset("lhoestq/conll2003")
print(raw_datasets)

def write_split(filename: str, split_name: str):
    """
    CoNLL-2003 format:
    """
    with open(filename, "w", encoding="utf-8") as f:
        for example in raw_datasets[split_name]:
            tokens = example["tokens"]
            tag_ids = example["ner_tags"]

            for tok, tag_id in zip(tokens, tag_ids):
                f.write(f"{tok} {tag_id}\n")
            f.write("\n")

write_split("./data/train.txt", "train")
write_split("./data/dev.txt", "validation")
write_split("./data/test.txt", "test")

print("Successfully get data and write to ./data/")
