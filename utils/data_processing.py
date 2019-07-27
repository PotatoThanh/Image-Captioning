import tensorflow as tf
import os
import json

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def download_COCOdata(save_folder):
    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                             cache_subdir=os.path.abspath(
                                                 save_folder),
                                             origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                             extract=True)

    annotation_file = os.path.dirname(
        annotation_zip)+'/annotations/captions_train2014.json'

    name_of_zip = 'train2014.zip'
    if not os.path.exists(os.path.abspath(save_folder) + '/' + name_of_zip):
        image_zip = tf.keras.utils.get_file(name_of_zip,
                                            cache_subdir=os.path.abspath(
                                                save_folder),
                                            origin='http://images.cocodataset.org/zips/train2014.zip',
                                            extract=True)
        data_path = os.path.dirname(image_zip)+'/train2014/'
    else:
        data_path = os.path.abspath(save_folder)+'/train2014/'

    return annotation_file, data_path

# pre-processing pair_imgname_caption


def gen_pair_imgname_caption(annotation_file, data_path):
    # Read the json file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Store captions and image names in vectors
    all_captions = []
    all_img_name_vector = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = data_path + \
            'COCO_train2014_' + '%012d.jpg' % (image_id)

        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    return all_captions, all_img_name_vector

# Load image encoded by InceptionV3


def load_image(image_path):
    # Preprocess the images using InceptionV3
    img = tf.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_images(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

# Find the maximum length of any caption in our dataset


def calc_max_length(tensor):
    return max(len(t) for t in tensor)

# Choose the top 5000 words from the vocabulary


def data_tokenizer(train_captions, top_k=5000):
    # top_k choosing the top 5000 words from the vocabulary
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    tokenizer.word_index['<pad>'] = 0

    # creating the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    # padding each vector to the max_length of the captions
    # if the max_length parameter is not provided, pad_sequences calculates that automatically
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(
        train_seqs, padding='post')

    # calculating the max_length
    # used to store the attention weights
    max_length = calc_max_length(train_seqs)
    return cap_vector, max_length, tokenizer

# Split the data into training and validation


def split_dataset(cap_vector, img_name_vector, test_size=0.2):
    # Create training and validation sets using 80-20 split
    cap_train, cap_val, img_name_train, img_name_val = train_test_split(cap_vector,
                                                                        img_name_vector,
                                                                        test_size=test_size,
                                                                        random_state=0)
    print('train-val: ' + str(len(img_name_train)),
          len(cap_train), len(img_name_val), len(cap_val))

    return cap_train, cap_val, img_name_train, img_name_val

# Shuffle imag_name and caption at the same time
def data_shuffle(all_captions, all_img_name_vector):
    # shuffling the captions and image_names together. setting a random state
    captions, img_name_vector = shuffle(all_captions,
                                       all_img_name_vector,
                                       random_state=1)
    return captions, img_name_vector
