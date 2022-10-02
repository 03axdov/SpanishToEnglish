import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import pathlib

class ShapeChecker():
  def __init__(self):
    # Keep a cache of every axis-name seen
    self.shapes = {}

  def __call__(self, tensor, names, broadcast=False):
    if not tf.executing_eagerly():
      return

    if isinstance(names, str):
      names = (names,)

    shape = tf.shape(tensor)
    rank = tf.rank(tensor)

    if rank != len(names):
      raise ValueError(f'Rank mismatch:\n'
                       f'    found {rank}: {shape.numpy()}\n'
                       f'    expected {len(names)}: {names}\n')

    for i, name in enumerate(names):
      if isinstance(name, int):
        old_dim = name
      else:
        old_dim = self.shapes.get(name, None)
      new_dim = shape[i]

      if (broadcast and new_dim == 1):
        continue

      if old_dim is None:
        # If the axis name is new, add its length to the cache.
        self.shapes[name] = new_dim
        continue

      if new_dim != old_dim:
        raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                         f"    found: {new_dim}\n"
                         f"    expected: {old_dim}\n")


def load_data(BATCH_SIZE=64, debug=False):
    path_to_zip = tf.keras.utils.get_file(
        "spa-eng.zip", origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
        extract=True
    )

    path_to_file = pathlib.Path(path_to_zip).parent/"spa-eng/spa.txt"
    text = path_to_file.read_text(encoding="utf-8")

    lines = text.splitlines()
    pairs = [line.split("\t") for line in lines]

    inp = [inp for targ, inp in pairs]
    targ = [targ for targ, inp in pairs]

    BUFFER_SIZE = len(inp)

    dataset = tf.data.Dataset.from_tensor_slices((inp, targ)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)

    if debug:
        for input_batch, target_batch in dataset.take(1):
            print(input_batch[:5])
            print("")
            print(target_batch[:5])
            break

    return dataset


if __name__ == "__main__":
    load_data(debug=True)