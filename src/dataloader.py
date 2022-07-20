from os.path import abspath, splitext
from typing import Optional

import numpy as np
from datasets import load_dataset, logging

logging.set_verbosity(logging.ERROR)

# Write code to load custom data.
def load(
    tokenizer,
    seq_len,
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    train_test_split: Optional[float] = None,
    worker: int = 1,
    batch_size: int = 1000,
    shuffle_seed: Optional[int] = None,
):
    TAG2NUM = {
        t: i
        for i, t in enumerate(
            [
                "O",
                "B-DT",
                "B-LC",
                "B-OG",
                "B-PS",
                "B-QT",
                "B-TI",
                "I-DT",
                "I-LC",
                "I-OG",
                "I-PS",
                "I-QT",
                "I-TI",
            ]
        )
    }

    def _processing(data):
        encoding = tokenizer(
            " ".join(eval(data["tokens"])),
            return_offsets_mapping=True,
            max_length=seq_len,
            truncation=True,
            padding="max_length",
        )

        label2 = [0]
        for idx, tag in enumerate(eval(data["ner_tags"])):
            if tag != "O":
                label2.extend([TAG2NUM["B-" + tag]])
                label2.extend(
                    [TAG2NUM["I-" + tag]] * (len(eval(data["tokens"])[idx]) - 1)
                )
            else:
                label2.extend([TAG2NUM["O"]] * len(eval(data["tokens"])[idx]))
            label2.extend([TAG2NUM["O"]])

        labels = np.zeros(len(encoding["offset_mapping"]), dtype=int)
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[1] != 0:
                labels[idx] = label2[mapping[0]]
        del encoding["offset_mapping"]
        encoding["labels"] = labels
        return encoding

    train_data_path = abspath(train_data_path)
    is_eval = False
    _, extention = splitext(train_data_path)

    datafiles = {"train": train_data_path}
    if eval_data_path is not None:
        assert (
            train_test_split is None
        ), "Only one of eval_data_path and train_test_split must be entered."
        datafiles["test"] = abspath(eval_data_path)
        is_eval = True

    if train_test_split is not None:
        assert (
            0.0 < train_test_split < 1.0
        ), "train_test_split must be a value between 0 and 1"
        train_test_split = int(train_test_split * 100)
        train_test_split = {
            "train": f"train[:{train_test_split}%]",
            "test": f"train[{train_test_split}%:]",
        }
        is_eval = True

    data = load_dataset(
        extention.replace(".", ""), data_files=datafiles, split=train_test_split
    )
    if shuffle_seed is not None:
        data = data.shuffle(seed=shuffle_seed)

    data = data.map(
        _processing,
        num_proc=worker,
        remove_columns=data["train"].column_names,
    )

    return (
        data["train"],
        data["test"] if is_eval else None,
    )


# Write preprocessor code to run in batches.
def default_collator(x, y):
    return x, y
