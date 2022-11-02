"""
Feature extraction with a pretrianed model.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

def load_pretrained_encoder():
  trillsson = hub.KerasLayer(
    "https://tfhub.dev/google/trillsson5/1", 
    trainable=False)
  inp = tf.keras.layers.Input((None,))
  out = trillsson(inp)["embedding"]
  model = tf.keras.Model(inp, out, name="encoder")
  return model

def prepare_example(waveform, label, sequence_length=16000):
  waveform = tf.cast(waveform, tf.float32) / float(tf.int16.max)
  padding = tf.maximum(sequence_length - tf.shape(waveform)[0], 0)
  left_pad = padding // 2
  right_pad = padding - left_pad
  waveform = tf.pad(waveform, paddings=[[left_pad, right_pad]])
  return waveform, label

if __name__ == "__main__":
  feature_directory = "./features/"
  if not os.path.exists(feature_directory):
    os.makedirs(feature_directory)
  dataset = "speech_commands"
  model_name = "trillsson"
  batch_size = 128
  autotune = tf.data.AUTOTUNE

  print("Loading speech_commands dataset...", flush=True)
  (ds_train, ds_val, ds_test), ds_info = tfds.load("speech_commands", 
    split=["train", "validation", "test"], shuffle_files=True, 
    as_supervised=True, with_info=True)
  num_classes =  ds_info.features["label"].num_classes

  ds_train = ds_train.map(prepare_example, num_parallel_calls=autotune)
  ds_train = ds_train.batch(batch_size).prefetch(autotune)

  ds_val = ds_val.map(prepare_example, num_parallel_calls=autotune)
  ds_val = ds_val.batch(batch_size).prefetch(autotune)

  ds_test = ds_test.map(prepare_example, num_parallel_calls=autotune)
  ds_test = ds_test.batch(batch_size).prefetch(autotune)

  print("Loading pretrained encoder...", flush=True)  
  model = load_pretrained_encoder()

  print("Computing features on training set...", flush=True)
  features, labels = [], []
  for i, (x,y) in enumerate(ds_train):
    f = model(x)
    features.append(f.numpy())
    labels.append(y.numpy())
  features = np.vstack(features)
  labels = np.concatenate(labels)
  np.savez(os.path.join(feature_directory, f"train_{dataset}"), 
    features=features, labels=labels)

  print("Computing features on validation set...", flush=True)
  features, labels = [], []
  for i, (x,y) in enumerate(ds_val):
    f = model(x)
    features.append(f.numpy())
    labels.append(y.numpy())
  features = np.vstack(features)
  labels = np.concatenate(labels)
  np.savez(os.path.join(feature_directory, f"validation_{dataset}"), 
    features=features, labels=labels)

  print("Computing features on testing set...", flush=True)
  features, labels = [], []
  for i, (x,y) in enumerate(ds_test):
    f = model(x)
    features.append(f)
    labels.append(y.numpy())
  features = np.vstack(features)
  labels = np.concatenate(labels)
  np.savez(os.path.join(feature_directory, f"test_{dataset}"), 
    features=features, labels=labels)

  print("Done.", flush=True)