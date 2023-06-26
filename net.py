
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from Bio.PDB import *
import numpy as np
import os
from tqdm import tqdm
import pathlib
import torch
from Bio.PDB import PDBParser, Polypeptide, is_aa
from esm import FastaBatchedDataset, pretrained
from keras.models import load_model

# so we can import utils notebook (delete if working on Pycharm), you might need to change it to your working directory path
# %cd /content/drive/MyDrive/Colab Notebooks/hackaton_2023_bio/
# import import_ipynb
import utils

###############################################################################
#                                                                             #
#              Parameters you can change, but don't have to                   #
#                                                                             #
###############################################################################


# number of ResNet blocks for the first ResNet and the kernel size.
RESNET_1_BLOCKS = 3
RESNET_1_KERNEL_SIZE = 15
RESNET_1_KERNEL_NUM = 64


###############################################################################
#                                                                             #
#                        Parameters you need to choose                        #
#                                                                             #
###############################################################################

RESNET_2_BLOCKS = 8
RESNET_2_KERNEL_SIZE = 30  # good start may be 3/5
RESNET_2_KERNEL_NUM = 64
DILATION = [1, 2, 4, 8, 16]

# percentage of dropout for the dropout layer
DROPOUT = 0.25 # good start may be 0.1-0.5

# number of epochs, Learning rate and Batch size
EPOCHS = 70
LR = 0.001 # good start may be 0.0001/0.001/0.01
BATCH = 32 # good start may be 32/64/128

CV_TIMES = 5

def resnet_1(input_layer):
    """
    ResNet layer - input -> BatchNormalization -> Conv1D -> Relu -> BatchNormalization -> Conv1D -> Relu -> Add
    :param input_layer: input layer for the ResNet
    :return: last layer of the ResNet
    """
    for i in range(RESNET_1_BLOCKS):
        batch_norm_layer = layers.BatchNormalization()(input_layer)
        conv1d_layer = layers.Conv1D(RESNET_1_KERNEL_NUM, RESNET_1_KERNEL_SIZE, activation='relu', padding='same')(batch_norm_layer)
        batch_norm_layer = layers.BatchNormalization()(conv1d_layer)
        input_layer = layers.Conv1D(RESNET_1_KERNEL_NUM, RESNET_1_KERNEL_SIZE, activation='relu', padding='same')(batch_norm_layer) + input_layer
    return input_layer

def resnet_2(input_layer):
    """
    Dilated ResNet layer - input -> BatchNormalization -> dilated Conv1D -> Relu -> BatchNormalization -> dilated Conv1D -> Relu -> Add
    :param input_layer: input layer for the ResNet
    :return: last layer of the ResNet
    """
    for i in range(RESNET_2_BLOCKS):
      for dilation_rate in DILATION:
          batch_norm_layer = layers.BatchNormalization()(input_layer)
          dilated_conv1d_layer = layers.Conv1D(RESNET_2_KERNEL_NUM, RESNET_2_KERNEL_SIZE, activation='relu', padding='same', dilation_rate=dilation_rate)(batch_norm_layer)
          batch_norm_layer = layers.BatchNormalization()(dilated_conv1d_layer)
          input_layer = layers.Conv1D(RESNET_2_KERNEL_NUM, RESNET_2_KERNEL_SIZE, activation='relu', padding='same', dilation_rate=dilation_rate)(batch_norm_layer) + input_layer
    return input_layer

def build_network(feature_num):
    input_layer = tf.keras.Input(shape=(utils.NB_MAX_LENGTH, feature_num))

    # Conv1D -> shape = (NB_MAX_LENGTH, RESNET_1_KERNEL_NUM)
    conv1d_layer = layers.Conv1D(RESNET_1_KERNEL_NUM, RESNET_1_KERNEL_SIZE, padding='same')(input_layer)

    # First ResNet -> shape = (NB_MAX_LENGTH, RESNET_1_KERNEL_NUM)
    resnet_layer = resnet_1(conv1d_layer)

    # Conv1D -> shape = (NB_MAX_LENGTH, RESNET_2_KERNEL_NUM)
    conv1d_layer = layers.Conv1D(RESNET_2_KERNEL_NUM, RESNET_2_KERNEL_SIZE, padding="same")(resnet_layer)

    # Second ResNet -> shape = (NB_MAX_LENGTH, RESNET_2_KERNEL_NUM)
    resnet_layer = resnet_2(conv1d_layer)

    # Apply self-attention
    attention_layer = scaled_dot_product_attention(resnet_layer)

    # Dropout layer -> shape = (NB_MAX_LENGTH, RESNET_2_KERNEL_NUM)
    dropout_layer = layers.Dropout(DROPOUT)(attention_layer)

    # Conv1D -> shape = (NB_MAX_LENGTH, RESNET_2_KERNEL_NUM/2)
    conv1d_layer = layers.Conv1D(RESNET_2_KERNEL_NUM // 2, RESNET_2_KERNEL_SIZE, padding="same")(dropout_layer)
    # Apply the Elu Activation Function after the 1D Convolution layer
    conv1d_layer = layers.Activation('elu')(conv1d_layer)

    # Dense layer -> shape = (NB_MAX_LENGTH, OUTPUT_SIZE)
    output_layer = layers.Dense(utils.OUTPUT_SIZE, activation='linear')(conv1d_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def scaled_dot_product_attention(inputs):
    q = layers.Conv1D(RESNET_2_KERNEL_NUM, kernel_size=1, padding="same")(inputs)
    k = layers.Conv1D(RESNET_2_KERNEL_NUM, kernel_size=1, padding="same")(inputs)
    v = layers.Conv1D(RESNET_2_KERNEL_NUM, kernel_size=1, padding="same")(inputs)

    # Calculate attention weights
    attention_weights = tf.matmul(q, k, transpose_b=True)
    attention_weights = tf.nn.softmax(attention_weights, axis=-1)

    # Apply attention weights to value
    attention_output = tf.matmul(attention_weights, v)

    # Residual connection and layer normalization
    attention_output += inputs
    attention_output = layers.LayerNormalization()(attention_output)

    return attention_output

def plot_val_train_loss(history, model_name):
    """
    plots the train and validation loss of the model at each epoch, saves it in 'model_loss_history.png'
    :param history: history object (output of fit function)
    :return: None
    """
    fig, axes = plt.subplots(1, 1, figsize=(15, 3))
    axes.plot(history.history['loss'], label='Training loss')
    axes.plot(history.history['val_loss'], label='Validation loss')
    axes.legend()
    axes.set_title("Train and Val MSE loss")
    axes.set_yscale('log')  # Set y-axis scale to logarithmic
    save_path = "/content/drive/MyDrive/Colab Notebooks/hackaton_2023_bio/Results/model_loss_history_{model_name}"
    plt.savefig(save_path)

    # ig, axes = plt.subplots(1, 1, figsize=(15,3))
    # axes.plot(history.history['loss'], label='Training loss')
    # axes.plot(history.history['val_loss'], label='Validation loss')
    # axes.legend()
    # axes.set_title("Train and Val MSE loss")
    # save_path = "/content/drive/MyDrive/Colab Notebooks/hackaton_2023_bio/Results/model_loss_history"
    # plt.savefig(save_path)

def extract_embeddings(model_name, protain_seq, protain_name):
    os.makedirs(utils.path_to_save_emmbending(model_name), exist_ok=True)
    repr_layers = [utils.LAYERS_NUMBER[model_name]]
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    print("done download")
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        batch_converter = alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter([(protain_name, protain_seq)])

        if torch.cuda.is_available():
            batch_tokens = batch_tokens.to(device="cuda", non_blocking=True)

        out = model(batch_tokens, repr_layers=repr_layers, return_contacts=False)

        representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
        for i, label in enumerate(batch_labels):
            entry_id = label.split()[0]

            filename = os.path.join(utils.path_to_save_emmbending(model_name), f"{entry_id}.pt")
            result = {"entry_id": entry_id}

            # save amino acid embeddings instead of mean representation
            result["amino_acid_embeddings"] = {layer: t[i, 1:-1].clone() for layer, t in representations.items()}
            torch.save(result, filename)


def generate_input(pt_file, model_name):
  repr_layers = utils.LAYERS_NUMBER[model_name]
  data = torch.load(pt_file)['amino_acid_embeddings'][repr_layers]
  padded_data= torch.zeros((140, utils.EMBENDING_DIM[model_name]))
  padded_data[:data.size(0),:] = data
  return padded_data.numpy()

def train_model(model_name):

  input_path = utils.input_path(model_name)
  X = np.array(np.load(input_path, allow_pickle=True))

  output_path = utils.output_path(model_name)
  Y = np.array(np.load(output_path, allow_pickle=True))

  X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

  # Compile model using Adam optimizer and MSE loss
  optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
  feature_num = X.shape[2]
  model = build_network(feature_num)
  model.compile(optimizer=optimizer, loss='mean_squared_error')

  history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=EPOCHS, batch_size=BATCH)
  model.save(utils.model_saved_path(model_name))
  print(f"save model at path {utils.model_saved_path(model_name)}")
  plot_val_train_loss(history, model_name)
  return model

def load_the_model(model_name):
  model = load_model(utils.model_saved_path(model_name))
  return model

def predict_model(model, model_name, seq_protain, protain_name):
  path_save_emmbending = utils.path_to_save_emmbending(model_name)
  extract_embeddings(model_name, seq_protain, protain_name)

  output_embedding = generate_input(f"{path_save_emmbending}/{protain_name}.pt", model_name)
  output_embedding_add_dimantion = np.expand_dims(output_embedding, axis=0)

  y_preds = model.predict(output_embedding_add_dimantion, batch_size=BATCH)
  return utils.matrix_to_pdb(seq_protain, y_preds[0], protain_name, model_name)

def main():
  for model_name in utils.MODELS_LIST:
    train_model(model_name)

def use_model(seq_protain, protain_name, model_name, train_again=False):
  if train_again or not os.path.exists(utils.model_saved_path(model_name)):
    model = train_model(model_name)
  else:
    model = load_the_model(model_name)
  return predict_model(model, model_name, seq_protain, protain_name)


# if __name__ == '__main__':
#     main()

# tests

def write_pdb_file_paths(directory, model_name):
    # model = load_the_model(model_name)
    path_model = f"/content/drive/MyDrive/Colab Notebooks/hackaton_2023_bio/models/saved_model_with_biger_kernel_{model_name}.h5"
    model = load_model(path_model)
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            seq, _ = utils.get_seq_aa(file_path, 'H')
            protain_name = os.path.splitext(os.path.basename(file_path))[0]
            predict_model(model, model_name, seq, protain_name)

# Example usage:
# model_name = 'esm2_t36_3B_UR50D'
# model_name = 'esm1b_t33_650M_UR50S'
# model_name = 'esm2_t6_8M_UR50D'
# model_name = 'one_hot'
# models = ['esm2_t36_3B_UR50D']
# directory = '/content/drive/MyDrive/Colab Notebooks/hackaton_2023_bio/tests'  # Specify the directory path
# for model_name in models:
#   write_pdb_file_paths(directory, model_name)