from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
import contractions
import numpy as np
import pandas as pd

en_model = spacy.load('en_core_web_md')


class ModelTrainer:
    def __init__(self, optimizer, loss_function, encoder_model, decoder_model, x_tokenizer, y_tokenizer):
        self.opt = optimizer
        self.loss_fn = loss_function
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.metrics = {'train_loss': [], 'val_loss': []}
        self.xtk = x_tokenizer
        self.ytk = y_tokenizer

    def train(self, training_data, validation_data, epochs, batch_size=32):

        self.x_maxlen = training_data[0].shape[-1]
        self.y_maxlen = training_data[1].shape[-1]

        training = tf.data.Dataset.from_tensor_slices(training_data).batch(batch_size)
        validation = tf.data.Dataset.from_tensor_slices(validation_data).batch(batch_size)

        for e in range(1, epochs+1):

            loss = 0
            print(f"epoch = {e}/{epochs} learning rate = {self.opt.learning_rate.numpy()}")
            for b,(x,y) in enumerate(tqdm(training)):
                if x.shape[0] == batch_size:
                    loss += self._train_step(x, y, batch_size).numpy()

            print('train_loss =', loss/b)
            training.shuffle(buffer_size=batch_size//4, reshuffle_each_iteration=True)
            self.metrics['train_loss'].append(loss/b)

            loss = 0
            for b,(x,y) in enumerate(validation):
                if x.shape[0] == batch_size:
                    loss += self._val_step(x, y, batch_size).numpy()

            print('val_loss =',loss/b)
            validation.shuffle(buffer_size=batch_size//4,reshuffle_each_iteration=True)
            self.metrics['val_loss'].append(loss/b)

            self._reduce_lr(e)
            self._save_best_weights(e)

            print()

    # callback functions
    def _reduce_lr(self, epoch):
        lr = self.opt.learning_rate.numpy()
        if epoch > 1 and self.metrics['val_loss'][-2] < self.metrics['val_loss'][-1]:
            self.opt.learning_rate.assign(lr/10)

    def _save_best_weights(self, epoch):
        if epoch > 1 and min(self.metrics['val_loss'][:-1]) > self.metrics['val_loss'][-1]:
            self.encoder.save_weights('en_weights')
            self.decoder.save_weights('dec_weights')
            print('best weights saved')
        elif epoch == 1:
            self.encoder.save_weights('en_weights')
            self.decoder.save_weights('dec_weights')
            print('best weights saved')

    @tf.function
    def _train_step(self, x, y, batch_size):
        loss = 0
        with tf.GradientTape() as tape:
            en_out, en_h, en_c = self.encoder(x, training=True)
            dec_h, dec_c = en_h, en_c
            dec_in = tf.expand_dims([self.ytk.word_index['<sos>']] * batch_size, 1)

            for t in range(1, y.shape[-1]):
                pred, dec_h, dec_c = self.decoder([dec_in, dec_h, dec_c, en_out], training=True)
                loss += self.loss_fn(y[:, t], pred)
                dec_in = tf.expand_dims(y[:, t], 1)

        b_loss = loss / int(y.shape[-1])
        weights = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(loss, weights)
        self.opt.apply_gradients(zip(grads, weights))
        return b_loss

    @tf.function
    def _val_step(self, x, y, batch_size):
        en_out, en_h, en_c = self.encoder(x, training=False)
        dec_h, dec_c = en_h, en_c
        dec_in = tf.expand_dims([self.ytk.word_index['<sos>']] * batch_size, 1)
        loss = 0
        for t in range(1, y.shape[-1]):
            pred, dec_h, dec_c = self.decoder([dec_in, dec_h, dec_c, en_out], training=False)
            loss += self.loss_fn(y[:, t], pred)
            dec_in = tf.expand_dims(y[:, t], 1)
        return loss / int(y.shape[-1])

    def predict(self, text, beam_size=3, factor=0.7):
        """
        function converts input text to a numerical sequence and uses beam search to
        output the most probable sequence

        :param text: input text
        :param beam_size: number of beams (beam_size=1 is greedy search)
        :param factor: beam normalization factor (0 to 1) default 0.7
        :return: output sequence
        """
        toks = text.lower().split()
        toks = [contractions.fix(t) for t in toks]
        toks = en_model(" ".join(toks))
        toks = [str(t) for t in toks if (not t.is_punct) and (str(t) != "'s")]
        text = " ".join(toks)

        seq = self.xtk.texts_to_sequences([text])
        seq = pad_sequences(seq,maxlen=self.x_maxlen,padding='post')[0]

        x = tf.expand_dims(seq, 0)
        en_out, en_h, en_c = self.encoder(x)

        dec_h, dec_c = en_h, en_c
        dec_in = tf.expand_dims([self.ytk.word_index['<sos>']], 1)

        pred, dec_h, dec_c = self.decoder([dec_in, dec_h, dec_c, en_out])

        probs = np.log(pred[0].numpy())

        top_idx = np.argpartition(probs, -beam_size)[-beam_size:]
        beams = [[x] for x in top_idx]
        score = [probs[x] for x in top_idx]
        states = [(dec_h, dec_c) for x in range(beam_size)]
        ended_beams = []

        while len(beams) > 0:
            new_states = []
            new_beams = []
            new_score = []
            top = []

            for i, idx in enumerate(beams):
                dec_in = tf.expand_dims([idx[-1]], 1)
                dec_h, dec_c = states[i]
                pred, dec_h, dec_c = self.decoder([dec_in, dec_h, dec_c, en_out])
                probs = np.log(pred[0].numpy())
                top_idx = np.argpartition(probs, -beam_size)[-beam_size:]

                for x in top_idx:
                    top.append([i, x, probs[x], (dec_h, dec_c)])

            top = sorted(top, key=lambda x: x[2], reverse=True)[:beam_size]

            for x in top:
                i = x[0]
                new = beams[i] + [x[1]]
                new_beams.append(new)
                new_score.append(score[i] + x[2])
                new_states.append(x[-1])

            beams = new_beams
            score = new_score
            states = new_states

            for i, x in enumerate(beams):
                if x[-1] == self.ytk.word_index['<eos>'] or len(x) == self.y_maxlen:
                    ended_beams.append([x[:-1], score[i]])

                    beams[i] = None
                    score[i] = None
                    states[i] = None

            score = [x for x in score if x is not None]
            states = [x for x in states if x is not None]
            beams = [x for x in beams if x is not None]

        out = [(x[0],x[1]/len(x[0])**factor) for x in ended_beams]
        out = sorted(out, key=lambda x: x[1], reverse=True)[0][0]
        return ' '.join(self.ytk.index_word[x] for x in out)

    def plot_training_logs(self):
        pd.DataFrame(self.metrics).plot()

