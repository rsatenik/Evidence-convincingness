from preprocessing import BertPreprocessing
from model import my_models
from preprocessing import import_data
import numpy as np
import pandas as pd
import random

max_length = 128  # Maximum length of input sentence to the model.
labels = [0, 1]


train_data, test_data, valid_data = import_data()
model, bert_model = my_models(max_length)
model.load_weights('./checkpoints/my_checkpoint')
model.evaluate(test_data, verbose=1)


def check_persuasiveness(evidence1, evidence2):
    evidence_pairs = np.array([[str(evidence1), str(evidence2)]])
    test_sequence = BertPreprocessing(
        evidence_pairs, labels=None, batch_size=1, shuffle=False, include_labels=False,
    )
    prob = model.predict(test_sequence)[0]
    idx = np.argmax(prob)
    prob = prob[idx]
    pred = labels[idx]
    return pred, prob


df = pd.read_csv('train.csv')
df.label = df.label-1
valid_df = df[:300]
sequence_index = random.randint(0, 300)  # randomly pick an index of evidences
print('-----------RESULT------------')
print('evidence1:', valid_df.evidence_1[sequence_index])
print('evidence2:', valid_df.evidence_2[sequence_index])
print('The evidence{num} is more convincing with {percentage} % probability'.format(
    num=check_persuasiveness(valid_df.evidence_1[sequence_index], valid_df.evidence_2[sequence_index])[0]+1,
    percentage=check_persuasiveness(valid_df.evidence_1[sequence_index], valid_df.evidence_2[sequence_index])[1]*100))
