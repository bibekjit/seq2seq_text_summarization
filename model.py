import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Bidirectional
from luong_attention import LuongAttention


class Encoder(tf.keras.models.Model):
    def __init__(self, units, emb_dim, voc, pretrained_embeddings=None):
        """
        Encoder part of the seq2seq model

        :param units: number of rnn units
        :param emb_dim: size of embedding dimension
        :param voc: X vocab size
        :param pretrained_embeddings: pretrained embedding weights (GloVe, word2vec etc)

        :returns: encoded sequence, states
        """
        self.units = units
        self.emb_dim = emb_dim
        self.voc = voc
        self.pretrained = pretrained_embeddings
        super().__init__()

    def build(self, input_shape):
        if self.pretrained is None:
            self.emb = Embedding(self.voc,
                                 self.emb_dim,
                                 mask_zero=True)
        else:
            self.emb = Embedding(self.voc,
                                 self.emb_dim,
                                 mask_zero=True,
                                 weights=self.pretrained,
                                 trainable=True)
        self.lstm = Bidirectional( LSTM(self.units,
                                        return_state=True,
                                        return_sequences=True,
                                        recurrent_activation="sigmoid"))
        super().build(input_shape)

    def call(self, x):
        x = self.emb(x)
        x, fh, fc, bh, bc = self.lstm(x)
        h = tf.concat([fh, bh], axis=-1)
        c = tf.concat([fc, bc], axis=-1)
        return x, h, c

    def summary(self, input_shape):
        i = Input(shape=(input_shape))
        o = self.call(i)
        print(tf.keras.models.Model(i, o).summary())


class Decoder(tf.keras.models.Model):
    def __init__(self, units, emb_dim, voc, pretrained_embeddings=None):

        """
        Decoder part of the seq2seq model

        :param units: number of rnn units
        :param emb_dim: size of embedding dimension
        :param voc: y vocab size
        :param pretrained_embeddings: pretrained embedding weights (GloVe, word2vec etc)

        :returns: decoded sequence, states
        """

        self.units = units
        self.emb_dim = emb_dim
        self.voc = voc
        self.pretrained = pretrained_embeddings
        super().__init__()

    def build(self, input_shape):
        if self.pretrained is None:
            self.emb = Embedding(self.voc,
                                 self.emb_dim,
                                 mask_zero=True)
        else:
            self.emb = Embedding(self.voc,
                                 self.emb_dim,
                                 mask_zero=True,
                                 weights=self.pretrained,
                                 trainable=True)
            
        self.lstm = LSTM(self.units,
                         return_state=True,
                         return_sequences=True,
                         recurrent_activation="sigmoid",
                         dropout=0.2)

        self.attn = LuongAttention(units=self.units, alignment='concat')
        self.dense = Dense(self.voc, activation='softmax')
        super().build(input_shape)

    def call(self, x):
        i, h, c, en_out = x
        cv = self.attn([h, en_out])
        cv = tf.expand_dims(cv, 1)
        i = self.emb(i)
        i = tf.concat([i, cv], axis=-1)
        i, h, c = self.lstm(i, initial_state=[h, c])
        i = tf.reshape(i, (-1, i.shape[-1]))
        i = self.dense(i)
        return i, h, c

    def summary(self, inputs):
        o = self.call(inputs)
        print(tf.keras.models.Model(inputs, o).summary())