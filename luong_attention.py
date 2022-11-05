import tensorflow as tf


class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units, alignment='dot'):
        """
        This attention layer is used in the decoder part of the seq2seq model.

        :param units: number of dense units (same as decoder rnn units)
        :param alignment: type of alignment to get the attention score (default : 'dot')
        """

        self.units = units
        self.type = alignment
        super().__init__()

    def build(self, input_shape):
        self.w = tf.keras.layers.Dense(self.units)
        self.u = tf.keras.layers.Dense(self.units)
        self.v = tf.keras.layers.Dense(1)
        super().build(input_shape)

    def call(self, x):
        """
        Implements attention mechanism and returns the context vector
        :param x: consists of the encoder hidden states (value) and previous decoder hidden state (query)
        :return: context vector
        """
        # q -> (batch,units)
        # v -> (batch, time steps, units)
        q, v = x

        # add time axis to query
        q = tf.expand_dims(q, 1)

        # calculate energy
        if self.type == 'concat':
            e = self.w(q) + self.u(v)
            e = self.v(tf.nn.tanh(e))

        elif self.type == 'general':
            e = tf.matmul(self.w(v), q, transpose_b=True)

        elif self.type == 'dot':
            e = q * v

        else:
            raise ValueError("not a valid alignment type")

        # get attention weights
        # then the weighted sum and return context vector
        aw = tf.nn.softmax(e, axis=1)
        cv = aw * v
        return tf.reduce_sum(cv, axis=1)