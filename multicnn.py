import tensorflow as tf
import keras
import keras.layers as layers

num_features = 3000
sequence_length = 300
embedding_dimension = 100
filter_sizes = [3, 4, 5]

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_features)

def convolution():
    inn = layers.Input(shape=(sequence_length, embedding_dimension, 1))
    cnns = []
    for size in filter_sizes:
        conv = layers.Conv2D(filters=64, kernel_size=(size, embedding_dimension),
                            strides=1, padding='valid', activation='relu')(inn)
        pool = layers.MaxPool2D(pool_size=(sequence_length - size + 1, 1), padding='valid')(conv)
        cnns.append(pool)
    outt = layers.concatenate(cnns)
    model = keras.Model(inputs=inn, outputs=outt)
    return model

def cnn_mulfilter():
    model = keras.sequential([
        layers.Embedding(input_dim=num_features, output_dim=embedding_dimension,
                        input_length=sequence_length),
        layers.Reshape((sequence_length, embedding_dimension, 1)),
        convolution(),
        layers.Flatten(),
        layers.Dense(10, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(), 
                loss=keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])
    return model

model = cnn_mulfilter()
model.summary()
hist = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1)