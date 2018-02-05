from keras import layers, models, optimizers

MAXLEN = 128
CHARS = 1000
HIDDEN_DIM = 128


def make_encoder() -> models.Sequential:
    model = models.Sequential(name='encoder')
    model.add(layers.TimeDistributed(layers.Dense(HIDDEN_DIM), input_shape=(MAXLEN, CHARS)))
    model.add(layers.LSTM(HIDDEN_DIM, return_sequences=True))
    model.add(layers.LSTM(HIDDEN_DIM, return_sequences=False))
    model.summary()
    return model


def make_decoder() -> models.Sequential:
    model = models.Sequential(name='decoder')
    model.add(layers.RepeatVector(MAXLEN, input_shape=(HIDDEN_DIM,)))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(CHARS)))
    model.add(layers.TimeDistributed(layers.Activation('softmax')))
    model.summary()
    return model


def make_encoder_decoder() -> models.Sequential:
    model = models.Sequential(name='encoder_decoder')
    model.add(make_encoder())
    model.add(make_decoder())

    opt = optimizers.Adam(clipvalue=1.0)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
