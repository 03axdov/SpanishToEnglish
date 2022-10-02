import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import typing
from typing import Any, Tuple

import tensorflow as tf
import tensorflow_text as tf_text

import matplotlib.pyplot as plt
import matplotlib.ticker

import pathlib

from utils import load_data, ShapeChecker


def main():
    dataset = load_data()


if __name__ == "__main__":
    main()