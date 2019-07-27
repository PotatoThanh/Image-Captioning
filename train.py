from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.enable_eager_execution()

from model import my_nets, losses, inceptionv3
from utils import data_processing, parallel_data_loading
import config
import pickle
import json
import time
import os
import numpy as np
import matplotlib.pyplot as plt

import argparse             
import os                   
import sys

def train(DATA_DIR='.', RESULT_DIR='.'):
    # Dowload data
    save_data_path = DATA_DIR
    annotation_file, data_path = data_processing.download_COCOdata(save_data_path)

    # Generate image_name and caption
    all_captions, all_img_name_vector = data_processing.gen_pair_imgname_caption(
        annotation_file, data_path)
    print('number of captions:' + str(len(all_captions)))

    # num_examples = 20
    # all_captions = all_captions[:num_examples]
    # all_img_name_vector = all_img_name_vector[:num_examples]

    # Tokenzing
    all_captions, max_length, tokenizer = data_processing.data_tokenizer(
        all_captions, top_k=10000)

    # Save tockenizer for evaluation part
    save_json = os.path.join(RESULT_DIR, 'tokenizer.pickle')
    with open(save_json, 'wb') as handle:
        pickle.dump(tokenizer, handle)
    
    # Shuffle dataset
    all_captions, all_img_name_vector = data_processing.data_shuffle(
        all_captions, all_img_name_vector)

    # Split train and validation
    cap_train, cap_val, img_name_train, img_name_val = data_processing.split_dataset(
        all_captions, all_img_name_vector, test_size=0.1)

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


    # Define optimization method
    optimizer = tf.train.AdamOptimizer(config.lr)

    # Defince decoder and encoder
    encoder = my_nets.CNN_Encoder(embedding_dim)
    decoder = my_nets.RNN_Decoder(embedding_dim, units, vocab_size)

    """## Checkpoint"""
    checkpoint_dir = "checkpoints"
    checkpoint_dir = os.path.join(RESULT_DIR, checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    tf.gfile.MkDir(checkpoint_dir)
    tf.gfile.MkDir(checkpoint_prefix)
    ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer=optimizer)
    ckpt_manager = tf.contrib.checkpoint.CheckpointManager(ckpt, checkpoint_prefix, max_to_keep=10)

    # Start current training
    # if ckpt_manager.latest_checkpoint:
    #     status = ckpt.restore(ckpt_manager.latest_checkpoint)
    #     start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

    """ Parallel load data set """
    dataset = parallel_data_loading.parallel_loading(img_name_train, cap_train, BUFFER_SIZE, BATCH_SIZE)

    ""  # Training """
    # adding this in a separate cell because if you run the training cell
    # many times, the loss_plot array will be reset
    loss_plot = []

    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0
        
        for (batch, (img_tensor, target)) in enumerate(dataset):
            loss = 0
            
            # initializing the hidden state for each batch
            # because the captions are not related from image to image
            hidden = decoder.reset_state(batch_size=target.shape[0])

            dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
            
            with tf.GradientTape() as tape:
                features = encoder(img_tensor)
                
                for i in range(1, target.shape[1]):
                    # passing the features through the decoder
                    predictions, hidden, _ = decoder(dec_input, features, hidden)

                    loss += losses.loss_function(target[:, i], predictions)
                    
                    # using teacher forcing
                    dec_input = tf.expand_dims(target[:, i], 1)
            
            total_loss += (loss / int(target.shape[1]))
            
            variables = encoder.trainable_variables + decoder.trainable_variables
            
            gradients = tape.gradient(loss, variables) 
            
            optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())
            
            if batch % 10 == 0:
                print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, 
                                                            batch, 
                                                            loss.numpy() / int(target.shape[1])))
        
        if epoch % 10 ==0:
            ckpt_manager.save()

        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / len(cap_train))
        
        print ('Epoch {} Loss {:.6f}'.format(epoch + 1, 
                                            total_loss/len(cap_train)))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        
        # save loss figure
        plt.plot(loss_plot)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Plot')
        fig_name = checkpoint_prefix + '/loss.png'
        plt.savefig(fig_name)

# This is the entry point of this module:
if __name__ == '__main__':
    train()
    print("START")
    print("Function call: " + str(sys.argv))

    print("Parse arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='$DATA_DIR', help='Directory with data')
    parser.add_argument('--result_dir', type=str, default='$RESULT_DIR', help='Directory with results')
    
    FLAGS, unparsed = parser.parse_known_args()
    print(str(FLAGS))
    print(str(unparsed))
    
        # Determine absolute paths of the input and output directories.
    # In the cases where the first character is '$', use the 
    # corresponding environment variables.
    if (FLAGS.data_dir[0] == '$'):
        DATA_DIR = os.environ[FLAGS.data_dir[1:]]
    else:
        DATA_DIR = FLAGS.data_dir

    if (FLAGS.result_dir[0] == '$'):
        RESULT_DIR = os.environ[FLAGS.result_dir[1:]]
    else:
        RESULT_DIR = FLAGS.result_dir

    train(DATA_DIR, RESULT_DIR)
    
    
    