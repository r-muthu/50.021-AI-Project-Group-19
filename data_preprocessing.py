import pandas as pd
import numpy as np
import random
import datasets
from transformers import AutoTokenizer

import pandas as pd
import numpy as np
import random
import datasets
from transformers import AutoTokenizer

class CustomDataset:
    def __init__(self, file_path, model_checkpoint, max_len=512, seed=42):
        self.file_path = file_path
        self.model_checkpoint = model_checkpoint
        self.max_len = max_len
        self.seed = seed

        # Set seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Load dataset
        self.df = pd.read_csv(self.file_path)
        self.df.rename(columns={'Content': 'text', 'Label': 'label'}, inplace=True)

        # Stratified sampling with shuffle inside each group
        self.df = (
            self.df
            .groupby('label', group_keys=False)
            .apply(lambda x: x.sample(n=min(50000, len(x)), random_state=self.seed))
            .sample(frac=1, random_state=self.seed)  # Shuffle the concatenated result
            .reset_index(drop=True)
        )

        # Final global shuffle
        self.df = self.df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)

    def preprocess_function(self, input_data):
        return self.tokenizer(input_data["text"], max_length=self.max_len, padding="max_length", truncation=True)

    def get_splits(self, test_size=0.2, val_size=0.5):
        # Convert to Hugging Face dataset
        dataset = datasets.Dataset.from_pandas(self.df)

        # Tokenize dataset
        encoded_dataset = dataset.map(self.preprocess_function, batched=True)

        # Train-test split (80-20)
        split_dataset = encoded_dataset.train_test_split(test_size=test_size, seed=self.seed)

        # Validation-test split (10-10)
        test_valid_split = split_dataset["test"].train_test_split(test_size=val_size, seed=self.seed)

        train_dataset = split_dataset["train"]
        val_dataset = test_valid_split["train"]
        test_dataset = test_valid_split["test"]

        return train_dataset, val_dataset, test_dataset

    def get_tokenizer(self):
        return self.tokenizer

"""
dataset = CustomDataset("HateSpeechDatasetBalanced.csv", "google/bert_uncased_L-2_H-128_A-2")
train_data, val_data, test_data = dataset.get_splits()

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")"""