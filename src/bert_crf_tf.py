# encoding=utf-8
import os
import glob
import collections
import tensorflow as tf
from src.bert import modeling, optimization, tokenization
from src.bert.modeling import InputFeatures, Processor
import pickle


def create_model(bert_config, is_training, input_ids, input_mask, tag2label, labels, seq_lens):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        use_one_hot_embeddings=False)
    bert_embedding = model.get_all_encoder_layers()[-1]
    logits_seq = tf.layers.dense(bert_embedding, len(tag2label))
    log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(logits_seq, labels, seq_lens)
    preds_seq, crf_scores = tf.contrib.crf.crf_decode(logits_seq, transition_matrix, seq_lens)
    with tf.variable_scope("loss"):
        per_example_loss = -log_likelihood / tf.cast(seq_lens, tf.float32)
        loss = tf.reduce_mean(per_example_loss)
        return loss, per_example_loss, preds_seq, crf_scores


def model_fn_builder(bert_config, init_checkpoint, tag2label, learning_rate=5e-5, num_train_steps=0,
                     num_warmup_steps=0):
    def model_fn(features, mode):
        input_ids, input_mask, label_ids, seq_lens = [features.get(k) for k in \
                                            ("input_ids", "input_mask", "label_ids", "seq_lens")]
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        total_loss, per_example_loss, logits, probabilities = create_model(
            bert_config, is_training, input_ids, input_mask, tag2label, label_ids, seq_lens)
        tvars = tf.trainable_variables()
        assignment_map, _ = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps)
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            accu = tf.metrics.accuracy(labels=label_ids, predictions=logits)
            loss = tf.metrics.mean(values=per_example_loss)
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss,
                                                     eval_metric_ops={"eval_accu": accu, "eval_loss": loss})
        else:
            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions={"prob": probabilities})
        return output_spec
    return model_fn


def convert_single_example(ex_index, example, max_length, tokenizer, tag2label):
    label_ids = []
    tokens = tokenizer.tokenize(example.text)
    tokens = ["[CLS]"] + tokens[:max_length - 2] + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    for tag in example.label:
        label_ids.append(tag2label[tag])
    label_ids.insert(0, tag2label['O'])
    while len(label_ids) < max_length:
        label_ids.append(tag2label['O'])
    while len(input_ids) < max_length:
        input_ids.append(0)
        input_mask.append(0)
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(tokens))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    return InputFeatures(input_ids=input_ids, input_mask=input_mask, label_ids=label_ids, seq_lens=example.seq_lens)


def file_based_convert_examples_to_features(examples, max_length, tokenizer, output_file, tag2label):
    writer = tf.python_io.TFRecordWriter(output_file)
    for ex_index, example in enumerate(examples):
        feature = convert_single_example(ex_index, example, max_length, tokenizer, tag2label)

        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["seq_lens"] = create_int_feature([feature.seq_lens])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder, batch_size):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "seq_lens": tf.FixedLenFeature([], tf.int64)
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn():
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat().shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))
        return d
    return input_fn


def dump_model_fn_builder(bert_config, init_checkpoint, tag2label):
    def model_fn(features, mode):
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        label_ids = features["label_ids"]
        seq_lens = features["seq_lens"]
        _, _, preds_seq, _ = create_model(bert_config, False, input_ids, input_mask, tag2label, label_ids, seq_lens)
        tvars = tf.trainable_variables()
        assignment_map, _ = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        export_outputs = {
            'predict': tf.estimator.export.PredictOutput(preds_seq)
        }
        output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=preds_seq, export_outputs=export_outputs)
        return output_spec
    return model_fn


def serving_input_receiver_fn(max_length):
    input_ids = tf.placeholder(shape=[None, max_length], dtype=tf.int32, name="input_ids")
    input_mask = tf.placeholder(shape=[None, max_length], dtype=tf.int32, name="input_mask")
    label_ids = tf.placeholder(shape=[None, max_length], dtype=tf.int32, name="label_ids")
    seq_lens = tf.placeholder(shape=[None], dtype=tf.int32, name='seq_lens')

    features = {"input_ids": input_ids, "input_mask": input_mask, "label_ids": label_ids, 'seq_lens': seq_lens}
    return tf.estimator.export.build_raw_serving_input_receiver_fn(features)


class BertCrf:

    def __init__(self, bert_path, train_path, eval_path, max_length, batch_size, save_path, learning_rate, epoch, save_checkpoints_steps, tf_config, tag2label=None, do_lower_case=True, warmup_ratio=.1):
        self.bert_path = bert_path
        self.train_path = train_path
        self.eval_path = eval_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.save_path = save_path
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.save_checkpoints_steps = save_checkpoints_steps
        self.tf_config = tf_config
        self.do_lower_case = do_lower_case
        self.warmup_ratio = warmup_ratio
        self.tokenizer = None
        self.model = None
        self.predictor = None
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
        self.save_config()

    def save_config(self):
        self.config = {}
        self.config['bert_path'] = self.bert_path
        self.config['train_path'] = self.train_path
        self.config['eval_path'] = self.eval_path
        self.config['max_length'] = self.max_length
        self.config['batch_size'] = self.batch_size
        self.config['save_path'] = self.save_path
        self.config['learning_rate'] = self.learning_rate
        self.config['epoch'] = self.epoch
        self.config['save_checkpoints_steps'] = self.save_checkpoints_steps
        self.config['do_lower_case'] = self.do_lower_case
        self.config['warmup_ratio'] = self.warmup_ratio
        self.config['tag2label'] = self.tag2label
        self.config['label2tag'] = self.label2tag

    def fit(self):
        tf.gfile.MakeDirs(self.save_path)
        processor = Processor()
        train_examples = processor.get_train_examples(self.train_path)
        num_train_steps = int(len(train_examples) / self.batch_size * self.epoch)
        num_warmup_steps = 0 if self.model else int(num_train_steps * self.warmup_ratio)
        if not self.model:
            init_checkpoint = os.path.join(self.bert_path, "bert_model.ckpt")
            bert_config_file = os.path.join(self.bert_path, "bert_config.json")
            bert_config = modeling.BertConfig.from_json_file(bert_config_file)
            model_fn = model_fn_builder(
                bert_config=bert_config,
                init_checkpoint=init_checkpoint,
                tag2label=self.tag2label,
                learning_rate=self.learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps)
            run_config = tf.estimator.RunConfig(
                model_dir=self.save_path,
                save_checkpoints_steps=self.save_checkpoints_steps,
                session_config=self.tf_config)
            self.model = tf.estimator.Estimator(
                model_fn=model_fn,
                config=run_config)
        vocab_file = os.path.join(self.bert_path, "vocab.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file,
            do_lower_case=self.do_lower_case)
        train_file = os.path.join(self.save_path, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, self.max_length, self.tokenizer, train_file, self.tag2label)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=self.max_length,
            is_training=True,
            drop_remainder=True,
            batch_size=self.batch_size)
        self.model.train(input_fn=train_input_fn, max_steps=num_train_steps)
        with open(os.path.join(self.save_path, "config"), "wb") as out:
            pickle.dump(self.config, out)

    def evaluate(self):
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        processor = Processor()
        eval_examples = processor.get_test_examples(self.eval_path)
        eval_file = os.path.join(self.save_path, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, self.max_length, self.tokenizer, eval_file, self.tag2label)
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=self.max_length,
            is_training=False,
            drop_remainder=False,
            batch_size=self.batch_size)
        return self.model.evaluate(input_fn=eval_input_fn, steps=None)

    def predict(self, text):
        tokens = self.tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens[:self.max_length - 2] + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        input_ids += [0] * (self.max_length - len(input_ids))
        input_mask += [0] * (self.max_length - len(input_mask))
        features = {"input_ids": [input_ids], "input_mask": [input_mask], "label_ids": [[-1] * len(input_ids)], "seq_lens": [len(input_ids)]}
        predict_results = self.predictor(features)["output"].tolist()[0][1: len(text) + 1]
        labels = []
        for i in predict_results:
            labels.append(self.label2tag[i])
        return labels

    def load(self, dir):
        assert os.path.exists(dir)
        with open(os.path.join(dir, "config"), "rb") as fin:
            self.config = pickle.load(fin)
        vocab_file = os.path.join(self.config['bert_path'], "vocab.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file,
            do_lower_case=self.config['do_lower_case'])
        saved_model = sorted(glob.glob(os.path.join(dir, "exported", "*")))[-1]
        self.predictor = tf.contrib.predictor.from_saved_model(saved_model)

    def save(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(os.path.join(dir, "config"), "wb") as out:
            pickle.dump(self.config, out)
        bert_config_file = os.path.join(self.bert_path, "bert_config.json")
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        predictor = tf.estimator.Estimator(
            model_fn=dump_model_fn_builder(
                bert_config=bert_config,
                init_checkpoint=self.save_path,
                tag2label=self.tag2label
            ),
            config=tf.estimator.RunConfig(model_dir=self.save_path))
        predictor.export_savedmodel(os.path.join(dir, "exported"),
                                    serving_input_receiver_fn(self.max_length))


if __name__ == "__main__":
    # tf配置
    os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4'
    tf_config = tf.ConfigProto(log_device_placement=True)
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # bert配置
    bert_cfg = {
        'bert_path': 'C:/数据/模型/chinese_L-12_H-768_A-12',
        'train_path': '../data/ner_train',
        'eval_path': '../data/ner_eval',
        'max_length': 128,
        'batch_size': 32,
        'save_path': '../model/bert',
        'learning_rate': 2e-5,
        'epoch': 3,
        'save_checkpoints_steps': 100,
        'tf_config': tf_config
    }
    model = BertCrf(**bert_cfg)
    model.fit()
    print(model.evaluate())
    model.save('../model/bert_crf')
    model = BertCrf(**bert_cfg)
    model.load('../model/bert_crf')
    results = model.predict('招商银行田惠宇行长在股东大会上致辞')
    print(results)