# encoding=utf-8
import os
import tensorflow as tf
from src.bert import modeling, tokenization
import random
import numpy as np


class BertCrf:

    def __init__(self, logger, train_path, eval_path, bert_path, max_length, batch_size, rate, epoch,
                 loss, tf_config, model_path, summary_path, tag2label=None):
        self.logger = logger
        self.train_path = train_path
        self.eval_path = eval_path
        self.bert_path = bert_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.rate = rate
        self.epoch = epoch
        self.loss = loss
        self.tf_config = tf_config
        self.model_path = model_path
        self.summary_path = summary_path
        vocab_file = os.path.join(self.bert_path, 'vocab.txt')
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
        self.predictor = None
        if tag2label is None:
            tag2label = {
                'O': 0,
                'B-com': 1,
                'I-com': 2,
                'B-pos': 3,
                'I-pos': 4
            }
        self.tag2label = tag2label
        self.label2tag = {}
        for key in self.tag2label:
            self.label2tag[self.tag2label[key]] = key

    def get_input_feature(self, data_path):
        data = []
        sententce = ''
        label = [self.tag2label['O']]  # 对应起始[cls]
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '\n' == line:
                    tokens = self.tokenizer.tokenize(sententce)
                    tokens = ['[CLS]'] + tokens[:self.max_length - 2] + ['[SEP]']
                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    input_mask = [1] * len(input_ids)
                    input_ids += [0] * (self.max_length - len(input_ids))
                    input_mask += [0] * (self.max_length - len(input_mask))
                    label += [self.tag2label['O']] * (self.max_length - len(label))
                    data.append([input_ids, input_mask, label])
                    sententce = ''
                    label = [self.tag2label['O']]
                else:
                    word, tag = line.replace('\n', '').split('\t')
                    sententce += word
                    label.append(self.tag2label[tag])
        return data

    def batch_yield(self, data, shuffle=False):
        if shuffle:
            random.shuffle(data)
        input_ids, input_mask, labels = [], [], []
        for (input_ids_, input_mask_, label_) in data:
            if len(input_ids) == self.batch_size:
                yield input_ids, input_mask, np.asarray(labels)
                input_ids, input_mask, labels = [], [], []
            input_ids.append(input_ids_)
            input_mask.append(input_mask_)
            labels.append(label_)
        if len(input_ids) != 0:
            yield input_ids, input_mask, np.asarray(labels)

    def model(self, input_ids, input_mask, seq_lens, labels):
        bert_config_file = os.path.join(self.bert_path, 'bert_config.json')
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        bert_model = modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=input_ids,
            input_mask=input_mask,
            use_one_hot_embeddings=False)
        bert_embedding = bert_model.get_sequence_output()
        logits_seq = tf.layers.dense(bert_embedding, len(self.tag2label))
        log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(logits_seq, labels, seq_lens)
        preds_seq, crf_scores = tf.contrib.crf.crf_decode(logits_seq, transition_matrix, seq_lens)
        return preds_seq, log_likelihood

    def fit(self):
        train_data = self.get_input_feature(self.train_path)
        init_checkpoint = os.path.join(self.bert_path, 'bert_model.ckpt')
        tvars = tf.trainable_variables()
        assignment_map, _ = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="inputs_ids")
        input_mask = tf.placeholder(shape=[None, None], dtype=tf.int32, name="inputs_mask")
        labels = tf.placeholder(tf.int32, [None, None], name='labels')
        inputs_seq_len = tf.reduce_sum(input_mask, axis=-1)
        preds_seq, log_likelihood = self.model(input_ids, input_mask, inputs_seq_len, labels)
        tf.add_to_collection('preds_seq', preds_seq)
        loss = -log_likelihood / tf.cast(inputs_seq_len, tf.float32)
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
                for step, (input_ids_batch, input_mask_batch, labels_batch) in enumerate(self.batch_yield(train_data)):
                    _, curr_loss = sess.run([train_op, loss],
                                            feed_dict={input_ids: input_ids_batch, input_mask: input_mask_batch, labels: labels_batch})
                    if step % 10 == 0:
                        self.logger.info('epoch:%d, batch: %d, current loss: %f' % (i, step + 1, curr_loss))
            saver.save(sess, self.model_path)
            tf.summary.FileWriter(self.summary_path, sess.graph)
            self.evaluate(sess, input_ids, input_mask, labels, preds_seq)

    def evaluate(self, sess, input_ids, input_mask, labels, preds_seq):
        eval_data = self.get_input_feature(self.eval_path)
        tp_com = 0  # 正类判定为正类
        fp_com = 0  # 负类判定为正类
        fn_com = 0  # 正类判定为负类
        tp_pos = 0  # 正类判定为正类
        fp_pos = 0  # 负类判定为正类
        fn_pos = 0  # 正类判定为负类
        for _, (input_ids_batch, input_mask_batch, labels_batch) in enumerate(self.batch_yield(eval_data)):
            preds = sess.run(preds_seq, feed_dict={input_ids: input_ids_batch, input_mask: input_mask_batch, labels: labels_batch})
            for i in range(len(preds)):
                pred = preds[i]
                label = labels_batch[i]
                true_com, true_pos = self.label2entity(label)
                pred_com, pred_pos = self.label2entity(pred)
                tp_com += len(true_com & pred_com)
                fp_com += len(pred_com - true_com)
                fn_com += len(true_com - pred_com)
                tp_pos += len(true_pos & pred_pos)
                fp_pos += len(pred_pos - true_pos)
                fn_pos += len(true_pos - pred_pos)
        recall_com = tp_com / (tp_com + fn_com)
        precision_com = tp_com / (tp_com + fp_com)
        f1_com = (2 * recall_com * precision_com) / (recall_com + precision_com)
        recall_pos = tp_pos / (tp_pos + fn_pos)
        precision_pos = tp_pos / (tp_pos + fp_pos)
        f1_pos = (2 * recall_pos * precision_pos) / (recall_pos + precision_pos)
        self.logger.info('eval company recall:' + str(recall_com) + ', eval company precision:' + str(precision_com)
                         + ', eval company f1:' + str(f1_com) + ', eval position recall:'
                         + str(recall_pos) + ', eval position precision:' + str(precision_pos) + ', eval position f1:' + str(f1_pos))

    def label2entity(self, label):
        '''
        将预测结果[0, 1, 2, 2, 2, 0, 3, 4]转成(com_1_4, pos_6_7),com公司实体，1是起始位置，4是结束位置，方便统计
        :param label:
        :return:
        '''
        com_set = set()
        pos_set = set()
        entity = ''
        count = 0
        while count < len(label):
            if 'B-com' == self.label2tag[int(label[count])]:
                entity += 'com_' + str(count)
                count += 1
                while count < len(label):
                    if 'I-com' == self.label2tag[int(label[count])] and count == len(label) - 1:
                        entity += '_' + str(count)
                        break
                    if 'I-com' != self.label2tag[int(label[count])]:
                        entity += '_' + str(count - 1)
                        break
                    count += 1
                s = entity.split('_')
                if 3 == len(s) and s[1] != s[2]:
                    com_set.add(entity)
                entity = ''
            elif 'B-pos' == self.label2tag[int(label[count])]:
                entity += 'pos_' + str(count)
                count += 1
                while count < len(label):
                    if 'I-pos' == self.label2tag[int(label[count])] and count == len(label) - 1:
                        entity += '_' + str(count)
                        break
                    if 'I-pos' != self.label2tag[int(label[count])]:
                        entity += '_' + str(count - 1)
                        break
                    count += 1
                s = entity.split('_')
                if 3 == len(s) and s[1] != s[2]:
                    pos_set.add(entity)
                entity = ''
            else:
                count += 1
        return com_set, pos_set

    def load(self, path):
        '''
        加载模型
        :param path:
        :return:
        '''
        self.pred_sess = tf.Session(config=self.tf_config)
        saver = tf.train.import_meta_graph(path + '/model.meta')
        saver.restore(self.pred_sess, tf.train.latest_checkpoint(path))
        graph = tf.get_default_graph()
        self.input_ids = graph.get_tensor_by_name('input_ids:0')
        self.input_mask = graph.get_tensor_by_name('input_mask:0')
        self.labels = graph.get_tensor_by_name('labels:0')
        self.preds_seq = tf.get_collection('preds_seq')

    def close(self):
        self.pred_sess.close()

    def _predict_text_process(self, text):
        label = []
        tokens = self.tokenizer.tokenize(text)
        tokens = ['[CLS]'] + tokens[:self.max_length - 2] + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        input_ids += [0] * (self.max_length - len(input_ids))
        input_mask += [0] * (self.max_length - len(input_mask))
        label += [self.tag2label['O']] * (self.max_length - len(label))
        return np.asarray([input_ids]), np.asarray([input_mask]), np.asarray([label])

    def predict(self, text):
        input_ids, input_mask, label = self._predict_text_process(text)
        pred, _ = self.pred_sess.run(self.preds_seq, feed_dict={self.input_ids: input_ids, self.input_mask: input_mask, self.labels: label})
        return pred