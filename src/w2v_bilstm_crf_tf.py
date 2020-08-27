# encoding=utf-8
import tensorflow as tf
import random
import numpy as np


class W2VBiLstmCrf:

    def __init__(self, logger, train_path, eval_path, w2v, max_len, batch_size, epoch, loss, rate, num_units,
                 tf_config, model_path, summary_path, embedding_dim=300, tag2label=None):
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
        self.tf_config = tf_config
        self.model_path = model_path
        self.summary_path = summary_path
        self.embedding_dim = embedding_dim
        if tag2label is None:
            tag2label = {
                "O": 0,
                "B": 1,
                "I": 2
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

    def model(self, seqs, seq_lens, labels):
        cell_fw = tf.nn.rnn_cell.LSTMCell(self.num_units)
        cell_bw = tf.nn.rnn_cell.LSTMCell(self.num_units)
        ((rnn_fw_outputs, rnn_bw_outputs),
         (rnn_fw_final_state, rnn_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=seqs,
            sequence_length=seq_lens,
            dtype=tf.float32
        )
        rnn_outputs = tf.add(rnn_fw_outputs, rnn_bw_outputs)
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
        preds_seq, log_likelihood = self.model(seqs, seq_lens, labels)
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
                    _, curr_loss = sess.run([train_op, loss], feed_dict={seqs: seqs_batch, seq_lens: seq_lens_batch, labels: labels_batch})
                    if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                        self.logger.info("epoch:%d, batch: %d, current loss: %f" % (i, step+1, curr_loss))
            saver.save(sess, self.model_path)
            tf.summary.FileWriter(self.summary_path, sess.graph)
            self.evaluate(sess, seqs, seq_lens, labels, preds_seq)

    def evaluate(self, sess, seqs, seq_lens, labels, preds_seq):
        eval_data = self.get_input_feature(self.eval_path)
        sum = 0
        total = 0
        for _, (seqs_batch, seq_lens_batch, labels_batch) in enumerate(self.batch_yield(eval_data)):
            pred = sess.run([preds_seq], feed_dict={seqs: seqs_batch, seq_lens: seq_lens_batch, labels: labels_batch})
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
            pred = self.pred_sess.run([self.preds_seq], feed_dict={self.seqs: seq_pred, self.seq_lens: seq_len_pred, self.labels: label_pred})
            predict_results.append(pred)
            predict_lens.append(seq_len_pred[0])
        return self._predict_result_process(predict_results, predict_lens)


if __name__ == '__main__':
    import os
    import logging
    from gensim.models import KeyedVectors
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    w2v = KeyedVectors.load('../model/w2v/w2v.model')
    wblc_cfg = {
        'logger': logging,
        'train_path': '../data/ner_train',
        'eval_path': '../data/ner_eval',
        'w2v': w2v,
        'max_len': 20,
        'batch_size': 64,
        'epoch': 50,
        'loss': 'sgd',
        'rate': 0.001,
        'num_units': 64,
        'tf_config': tf_config,
        'model_path': '../model/w2v_bilstm_crf/model',
        'summary_path': '../model/w2v_bilstm_crf/summary'
    }
    model = W2VBiLstmCrf(**wblc_cfg)
    model.fit()
    model.load('../model/w2v_bilstm_crf')
    print(model.predict(['北京很大', '中国很大']))
    model.close()