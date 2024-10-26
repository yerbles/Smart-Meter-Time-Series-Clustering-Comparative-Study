# Module Description
# These functions allow the computation of alternative data representations using neural networks

# *******************
# Module Requirements
# *******************
# General
import numpy as np
from sklearn import preprocessing

# Tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Input, Model
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import RepeatVector, TimeDistributed, Dense

#########################
# Computing LSTM Features
#########################
def repLSTMae(data, node_arrangement, num_training_epochs, batch_size, checkpoint_filepath):
    """
    Top level function for building and training LSTM-based autoencoders. 
    Returns features from the encoder component after training, using the best model found during training.

    Parameters
    ----------
    data: array (shape = (num_ts, num_features))
        The data with time series or their alternative representations stored in the rows.

    node_arrangement: string
        One of "A" through "P". Refers to the node counts contained in the returnLayerNodeCount_LSTM() function.

    num_training_epochs: integer
        Number of epochs for training. Note that the best model is kept from all epochs.

    batch_size: integer
        Batch sizes for training the autoencoder

    checkpoint_filepath: string
        Combined directory and filename for a model checkpoint file

    Output
    ------
    features: array (shape=(num_ts, num_new_features))
    """
    num_layers = int((len(returnLayerNodeCount_LSTM(node_arrangement))-1)/2)
    AER = createCudnnModel_LSTM(node_arrangement)
    AER.summary()

    MC = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch")

    history = AER.fit(data, data, epochs=num_training_epochs, batch_size=batch_size, verbose=1, callbacks = [MC])

    AER = createCudnnModel_LSTM(node_arrangement)
    AER.load_weights(checkpoint_filepath)
    AER.evaluate(data, data, verbose=1)

    ER = Model(AER.layers[0].input,AER.layers[num_layers+1].output)
    ER.summary()

    features = ER.predict(data, verbose=1)
    print("Encoded Features - Shape: ", features.shape)

    return features


def returnLayerNodeCount_LSTM(which_option):
    layer_node_count = {
        # 1 layer
        "A": [48,36,48],
        "B": [48,24,48],
        "C": [48,12,48],
        "D": [36,24,36],
        "E": [36,12,36],
        "F": [24,12,24],
        # 2 layers
        "G": [48,36,24,36,48],
        "H": [48,36,12,36,48],
        "I": [48,24,12,24,48],
        "J": [36,24,12,24,36],
        # 3 layers
        "K": [48,36,24,12,24,36,48],
        # Random Others that we may jettison
        "L": [64,32,16,32,64],
        "M": [64,48,24,12,24,48,64],
        "N": [128,64,32,16,32,64,128],
        "O": [128,64,48,32,24,32,48,64,128],
        "P": [96,72,48,24,12,24,48,72,96]
    }
    return layer_node_count[which_option]


def createCudnnModel_LSTM(which_option):
    # Optimizer: Adam is an adaptive optimizer which requires no focus on a learning rate - best automatic option.
    # https://towardsdatascience.com/7-tips-to-choose-the-best-optimizer-47bb9c1219e
    # Loss Functions: CuDNNLSTM restricts the activation function of layers to tanh
    # Time-constraints for not playing around with other optimizers/loss functions - CuDNNLSTM takes advantage of GPUs

    layer_nodes = returnLayerNodeCount_LSTM(which_option)
    layer_count = 0
    num_outer_layers = int((len(layer_nodes)-1)/2)

    # Encoder
    inp = Input(shape=(48,1))

    en = CuDNNLSTM(layer_nodes[layer_count], return_sequences = True)(inp)
    layer_count += 1
    for layer in range(num_outer_layers-1):
            en = CuDNNLSTM(layer_nodes[layer_count], return_sequences = True)(en)
            layer_count += 1

    bottleneck = CuDNNLSTM(layer_nodes[layer_count])(en)
    layer_count += 1

    de = RepeatVector(48)(bottleneck)
    for layer in range(num_outer_layers):
            de = CuDNNLSTM(layer_nodes[layer_count], return_sequences=True)(de)
            layer_count += 1

    out = TimeDistributed(Dense(1))(de)

    # Autoencoder
    AER = Model(inp, out)
    AER.compile(optimizer='adam', loss='mse')

    return AER






# Test code
if __name__ == "__main__":
    x = np.random.normal(size=(50,48))
    scaler = preprocessing.MinMaxScaler()
    x = scaler.fit_transform(x)
    print("x.shape: ", x.shape)
    print("")

    # LSTM
    print("LSTM")
    features = repLSTMae(
        data = x, node_arrangement = "A", num_training_epochs = 100, batch_size = 10, 
        checkpoint_filepath = "/scratch3/yer002/filestorage/ComparativeStudy/model_checkpoints/A_100_10/cp.ckpt"
        )
    print("features.shape: ", features.shape)
