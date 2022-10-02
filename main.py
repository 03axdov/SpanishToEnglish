import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import typing
from typing import Any, Tuple

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker

import pathlib

from utils import load_data, tf_lower_and_split_punct
from models import Encoder, BahdanauAttention


def main():
    dataset, inp, targ = load_data()

    max_vocab_size = 5000

    input_text_processor = tf.keras.layers.TextVectorization(
        standardize=tf_lower_and_split_punct,
        max_tokens=max_vocab_size
    )

    input_text_processor.adapt(inp)
    # print(input_text_processor.get_vocabulary()[:10])

    output_text_processor = tf.keras.layers.TextVectorization(
        standardize=tf_lower_and_split_punct,
        max_tokens=max_vocab_size
    )

    output_text_processor.adapt(targ)
    # print(output_text_processor.get_vocabulary()[:10])

    embedding_dim = 256
    units = 1024

    model = Encoder(
        input_text_processor.vocabulary_size(),
        embedding_dim,
        units
    )

    attention_layer = BahdanauAttention(units)



if __name__ == "__main__":
    main()