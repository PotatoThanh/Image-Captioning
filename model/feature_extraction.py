import tensorflow as tf

def InceptionV3(trainable=False):
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    image_features_extract_model.trainable = trainable

    return image_features_extract_model

def MobileNetV2(trainable=False):
    image_model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=None,
                                                include_top=False, weights='imagenet')                                              
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    image_features_extract_model.trainable = trainable

    return image_features_extract_model
def simpleCNN(trainable=True):
    img = tf.keras.layers.Input(shape=(299, 299, 3))
    x = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=2, activation='relu', padding='same')(img)
    x = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    model = tf.keras.Model(inputs=[img],outputs=[x])
    model.trainable=trainable
    return model
def feature_extractor(model_type='simple', trainable=False):
    if model_type == 'inceptionv3':
        return InceptionV3(trainable=trainable)

    elif model_type == 'simple':
        return simpleCNN(trainable=trainable)