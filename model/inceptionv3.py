import tensorflow as tf

def InceptionV3(trainable=False):
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    image_features_extract_model.trainable = trainable

    return image_features_extract_model