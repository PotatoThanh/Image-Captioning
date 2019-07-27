# Our images and captions are ready! Next, let's create a tf.data dataset to use for training our model.
# feel free to change these parameters according to your system's configuration
BATCH_SIZE = 32
EPOCHS=300
BUFFER_SIZE = 1000
embedding_dim = 512
units = 1024
lr = 1e-3
# shape of the vector extracted from InceptionV3 is (64, 2048)
# these two variables represent that
features_shape = 2048
attention_features_shape = 64