##%%
# Import TensorFlow and enable eager execution
# This code requires TensorFlow version >=1.9
import tensorflow as tf
tf.enable_eager_execution()

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage

from utils import data_processing
from model.my_nets import CNN_Encoder, RNN_Decoder
from model.losses import loss_function
import config

def evaluate_func(image_name, max_length, RESULT_DIR):
    # Get tokenizer
    # loading
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Our images and captions are ready! Next, let's create a tf.data dataset to use for training our model.
    # feel free to change these parameters according to your system's configuration
    BATCH_SIZE = config.BATCH_SIZE
    EPOCHS = config.EPOCHS
    BUFFER_SIZE = config.BUFFER_SIZE
    embedding_dim = config.embedding_dim
    units = config.units
    vocab_size = len(tokenizer.word_index)

    # shape of the vector extracted from InceptionV3 is (64, 2048)
    # these two variables represent that
    features_shape = config.features_shape
    attention_features_shape = config.attention_features_shape
    attention_plot = np.zeros((attention_features_shape))

    # Create encoder and decoder 
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    # define optimizer
    optimizer = tf.train.AdamOptimizer()
    # Load checkpoint
    checkpoint_dir = "checkpoints"
    checkpoint_dir = os.path.join(RESULT_DIR, checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer=optimizer)
    ckpt_manager = tf.contrib.checkpoint.CheckpointManager(ckpt, checkpoint_prefix, max_to_keep=10)
    
    # Restore values from the checkpoint
    status = ckpt.restore(ckpt_manager.latest_checkpoint)
        
    # reset hidden state
    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(data_processing.load_image(image_name)[0], 0)
    features = encoder(temp_input)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []
    prob = []

    attention_plot= np.zeros((max_length, 64))
    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_prob = tf.reduce_max(tf.nn.softmax(predictions)).numpy()
        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])
        prob.append(predicted_prob)

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, prob, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)
        
    attention_plot = attention_plot[:len(result), :]
    return result, prob, attention_plot

def plot_attention(image_path, result, prob, attention_plot, smooth=True):
    # temp_image = np.array(Image.open(image_path))
    temp_image = tf.read_file(image_path)
    temp_image = tf.image.decode_jpeg(temp_image, channels=3)

    len_result = len(result)-1
    w = round(np.math.sqrt(len_result))
    h = np.math.ceil(len_result / w)

    fig = plt.figure(figsize=(w, h))

    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        if smooth == True:
            temp_att = skimage.transform.pyramid_expand(temp_att, upscale=32, sigma=20)

        ax = fig.add_subplot(w, h, l+1)
        ax.axis('off')
        ax.set_title(result[l]+'({0:.3f})'.format(prob[l]))

        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.show()


def evaluate(image_path, max_length, RESULT_DIR):
    result, prob, attention_plot = evaluate_func(image_path, max_length, RESULT_DIR)
    print ('Prediction Caption:', ' '.join(result))
    plot_attention(image_path, result, prob, attention_plot)

# This is the entry point of this module:
if __name__ == '__main__':
    image_path = 'train2014/COCO_train2014_000000000009.jpg'
    evaluate(image_path, max_length=29, RESULT_DIR='.')
    # print("START")
    # print("Function call: " + str(sys.argv))

    # print("Parse arguments...")
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', type=str, default='$DATA_DIR', help='Directory with data')
    # parser.add_argument('--result_dir', type=str, default='$RESULT_DIR', help='Directory with results')
    
    # FLAGS, unparsed = parser.parse_known_args()
    # print(str(FLAGS))
    # print(str(unparsed))
    
    #     # Determine absolute paths of the input and output directories.
    # # In the cases where the first character is '$', use the 
    # # corresponding environment variables.
    # if (FLAGS.data_dir[0] == '$'):
    #     DATA_DIR = os.environ[FLAGS.data_dir[1:]]
    # else:
    #     DATA_DIR = FLAGS.data_dir

    # if (FLAGS.result_dir[0] == '$'):
    #     RESULT_DIR = os.environ[FLAGS.result_dir[1:]]
    # else:
    #     RESULT_DIR = FLAGS.result_dir

    # evaluate(image_path, max_length, RESULT_DIR)