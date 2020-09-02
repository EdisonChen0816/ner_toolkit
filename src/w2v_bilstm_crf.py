# encoding=utf-8
import tensorflow as tf
import random
import numpy as np


class W2VBiLstmCrf:

    def __init__(self, logger, train_path, eval_path, w2v, max_len, batch_size, epoch, loss, rate, num_units,
                 num_layers, tf_config, model_path, summary_path, embedding_dim=300, tag2label=None, use_attention=False):
        self.logger = logger
        self.train_path = train_path
        self.eval_path = eval_path
        self.w2v = w2v
        self.max_len = max_len
        self.batch_size = batch_size
        self.epoch = epoch
        self.loss = loss
        self.rate = rate
        self.num_units = num_units
        self.num_layers = num_layers
        self.tf_config = tf_config
        self.model_path = model_path
        self.summary_path = summary_path
        self.embedding_dim = embedding_dim
        self.use_attention = use_attention
        if tag2label is None:
            tag2label = {
                "O": 0,
                "B-com": 1,
                "I-com": 2,
                "B-pos": 3,
                "I-pos": 4
            }
        self.tag2label = tag2label
        self.label2tag = {}
        for key in self.tag2label:
            self.label2tag[self.tag2label[key]] = key
        self.pred_sess = None

    def get_input_feature(self, data_path):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            seq = []
            label = []
            for line in f:
                if '\n' == line:
                    if len(label) != len(seq):
                        raise('label and seq, length is not match')
                    seq_len = len(seq)
                    if seq_len > self.max_len:
                        seq = seq[: self.max_len]
                        label = label[: self.max_len]
                    else:
                        for i in range(self.max_len - seq_len):
                            seq.append([0] * 300)
                            label.append(self.tag2label['O'])
                    data.append([seq, seq_len, label])
                    seq = []
                    label = []
                else:
                    word, tag = line.replace('\n', '').split('\t')
                    if word not in self.w2v:
                        word = 'unknown'
                    seq.append(self.w2v[word])
                    label.append(self.tag2label[tag])
            if len(seq) > 0 and len(seq) == len(label):
                seq_len = len(seq)
                if seq_len > self.max_len:
                    seq = seq[: self.max_len]
                    label = label[: self.max_len]
                    seq_len = self.max_len
                else:
                    for i in range(self.max_len - seq_len):
                        seq.append([0] * 300)
                        label.append(self.tag2label['O'])
                data.append([seq, seq_len, label])
        return np.asarray(data)

    def batch_yield(self, data, shuffle=False):
        if shuffle:
            random.shuffle(data)
        seqs, seq_lens, labels = [], [], []
        for (seq, seq_len, label) in data:
            if len(seqs) == self.batch_size:
                yield np.asarray(seqs), np.asarray(seq_lens), np.asarray(labels)
                seqs, seq_lens, labels = [], [], []
            seqs.append(seq)
            seq_lens.append(seq_len)
            labels.append(label)
        if len(seqs) != 0:
            yield np.asarray(seqs), np.asarray(seq_lens), np.asarray(labels)

    def attention(self, inputs, attention_size=128):
        hidden_size = inputs.shape[2].value
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')
        alphas = tf.nn.softmax(vu, name='alphas')
        output = tf.multiply(inputs, tf.expand_dims(alphas, -1))
        output = tf.add(output, inputs)
        return output

    def model(self, seqs, seq_lens, labels, keep_prob):
        cell_fws = []
        cell_bws = []
        for _ in range(self.num_layers):
            cell_fw = tf.contrib.rnn.BasicLSTMCell(self.num_units, forget_bias=1.0, state_is_tuple=True)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell=cell_fw, input_keep_prob=1.0, output_keep_prob=keep_prob,
                                                    state_keep_prob=1.0)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(self.num_units, forget_bias=1.0, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell=cell_bw, input_keep_prob=1.0, output_keep_prob=keep_prob,
                                                    state_keep_prob=1.0)
            cell_fws.append(cell_fw)
            cell_bws.append(cell_bw)
        stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell(cell_fws)
        stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell(cell_bws)
        ((rnn_fw_outputs, rnn_bw_outputs),
         (rnn_fw_final_state, rnn_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=stacked_lstm_fw,
            cell_bw=stacked_lstm_bw,
            inputs=seqs,
            sequence_length=seq_lens,
            dtype=tf.float32
        )
        rnn_outputs = tf.add(rnn_fw_outputs, rnn_bw_outputs)
        if self.use_attention:
            rnn_outputs = self.attention(rnn_outputs)
        logits_seq = tf.layers.dense(rnn_outputs, len(self.tag2label))
        log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(logits_seq, labels, seq_lens)
        preds_seq, crf_scores = tf.contrib.crf.crf_decode(logits_seq, transition_matrix, seq_lens)
        return preds_seq, log_likelihood

    def fit(self):
        train_data = self.get_input_feature(self.train_path)
        num_batches = (len(train_data) + self.batch_size - 1) // self.batch_size
        seqs = tf.placeholder(tf.float32, [None, self.max_len, 300], name="seqs")
        seq_lens = tf.placeholder(tf.int32, [None], name="seq_lens")
        labels = tf.placeholder(tf.int32, [None, None], name='labels')
        keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
        preds_seq, log_likelihood = self.model(seqs, seq_lens, labels, keep_prob)
        tf.add_to_collection("preds_seq", preds_seq)
        loss = -log_likelihood / tf.cast(seq_lens, tf.float32)
        loss = tf.reduce_mean(loss)
        if 'sgd' == self.loss.lower():
            train_op = tf.train.GradientDescentOptimizer(self.rate).minimize(loss)
        elif 'adam' == self.loss.lower():
            train_op = tf.train.AdamOptimizer(self.rate).minimize(loss)
        else:
            train_op = tf.train.GradientDescentOptimizer(self.rate).minimize(loss)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=self.tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epoch):
                for step, (seqs_batch, seq_lens_batch, labels_batch) in enumerate(self.batch_yield(train_data)):
                    _, curr_loss = sess.run([train_op, loss], feed_dict={seqs: seqs_batch, seq_lens: seq_lens_batch, labels: labels_batch, keep_prob: 1.0})
                    if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                        self.logger.info("epoch:%d, batch: %d, current loss: %f" % (i, step+1, curr_loss))
            saver.save(sess, self.model_path)
            tf.summary.FileWriter(self.summary_path, sess.graph)
            self.evaluate(sess, seqs, seq_lens, labels, keep_prob, preds_seq)

    def evaluate(self, sess, seqs, seq_lens, labels, keep_prob, preds_seq):
        eval_data = self.get_input_feature(self.eval_path)
        sum = 0
        total = 0
        for _, (seqs_batch, seq_lens_batch, labels_batch) in enumerate(self.batch_yield(eval_data)):
            pred = sess.run([preds_seq], feed_dict={seqs: seqs_batch, seq_lens: seq_lens_batch, labels: labels_batch, keep_prob: 1.0})
            sum += np.sum(pred == labels_batch)
            total += len(np.reshape(labels_batch, [-1]))
        self.logger.info('eval acc:' + str(sum / total))

    def load(self, path):
        self.pred_sess = tf.Session(config=self.tf_config)
        saver = tf.train.import_meta_graph(path + '/model.meta')
        saver.restore(self.pred_sess, tf.train.latest_checkpoint(path))
        graph = tf.get_default_graph()
        self.seqs = graph.get_tensor_by_name('seqs:0')
        self.labels = graph.get_tensor_by_name('labels:0')
        self.seq_lens = graph.get_tensor_by_name('seq_lens:0')
        self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
        self.preds_seq = tf.get_collection('preds_seq')

    def close(self):
        self.pred_sess.close()

    def _predict_result_process(self, predict_results, predict_lens):
        ners = []
        for i in range(len(predict_results)):
            tags = predict_results[i][0][0][0][: predict_lens[i]]
            ner = []
            for t in tags:
                ner.append(self.label2tag[t])
            ners.append(ner)
        return ners

    def _predict_text_process(self, text):
        seq = []
        label = []
        for word in list(text):
            if word not in self.w2v:
                word = 'unknown'
            seq.append(self.w2v[word])
            label.append(-1)
        seq_len = len(seq)
        if seq_len > self.max_len:
            seq = seq[: self.max_len]
            label = label[: self.max_len]
        else:
            for i in range(self.max_len - seq_len):
                seq.append([0] * 300)
                label.append(self.tag2label['O'])
        return np.asarray([seq]), np.asarray([seq_len]), np.asarray([label])

    def predict(self, texts):
        predict_results = []
        predict_lens = []
        for text in texts:
            seq_pred, seq_len_pred, label_pred = self._predict_text_process(text)
            pred = self.pred_sess.run([self.preds_seq], feed_dict={self.seqs: seq_pred, self.seq_lens: seq_len_pred, self.labels: label_pred, self.keep_prob: 1.0})
            predict_results.append(pred)
            predict_lens.append(seq_len_pred[0])
        return self._predict_result_process(predict_results, predict_lens)