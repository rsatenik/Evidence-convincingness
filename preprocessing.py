import tensorflow as tf
import pandas as pd
import transformers
import numpy as np

max_length = 128  # Maximum length of input sentence
batch_size = 32

labels = [0, 1]  # Labels in our dataset
df = pd.read_csv('train.csv')  # load train dataset
df.label = df.label-1  # change labels 1 and 2 to 0 and 1

train_df = df[300:]  # split train dataset into training and validation datasets
valid_df = df[:300]
test_df = pd.read_csv('test.csv')  # load test dataset
test_df.label = test_df.label-1 # change labels 1 and 2 to '0' and '1'

# Shape of the data
print("Load our data...")
print(f"Total train samples : {train_df.shape[0]}")
print(f"Total validation samples: {valid_df.shape[0]}")
print(f"Total test samples: {test_df.shape[0]}")

# convert labels to binary class matrix
y_train = tf.keras.utils.to_categorical(train_df.label, num_classes=2)
y_val = tf.keras.utils.to_categorical(valid_df.label, num_classes=2)
y_test = tf.keras.utils.to_categorical(test_df.label, num_classes=2)


class BertPreprocessing(tf.keras.utils.Sequence):
    """Generates batches of data.

    Arguments:
        evidence_pairs: Array of input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_labels: boolean, whether to include the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_labels=False`)
    """

    def __init__(
        self,
        evidence_pairs,
        labels,
        batch_size=batch_size,
        shuffle=True,
        include_labels=True,
    ):
        self.evidence_pairs = evidence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_labels = include_labels
        # Load our BERT Tokenizer to encode the sequences
        # We will use bart-base-uncased pretrained model
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.evidence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.evidence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        evidence_pairs = self.evidence_pairs[indexes]

        # With batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token
        encoded = self.tokenizer.batch_encode_plus(
            evidence_pairs.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        if self.include_labels:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


def import_data():
    train_data = BertPreprocessing(
        train_df[["evidence_1", "evidence_2"]].values.astype("str"),
        y_train,
        batch_size=batch_size,
        shuffle=True,
    )

    valid_data = BertPreprocessing(
        valid_df[["evidence_1", "evidence_2"]].values.astype("str"),
        y_val,
        batch_size=batch_size,
        shuffle=False,
    )

    test_data = BertPreprocessing(
        test_df[["evidence_1", "evidence_2"]].values.astype("str"),
        y_test,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_data, test_data, valid_data
