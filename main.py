# encoding=utf-8
from src.ac_match import ACMatch
from src.util.logger import setlogger
from src.util.yaml_util import loadyaml
from src.bilstm_crf import BiLstmCrf
import os
import tensorflow as tf
from gensim.models import KeyedVectors
from src.w2v_bilstm_crf import W2VBiLstmCrf
from src.bert_crf import BertCrf
from src.bert_bilstm_crf import BertBiLstmCrf


config = loadyaml('conf/NER.yaml')
logger = setlogger(config)

# tf配置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8


def test_ac_match(text):
    '''
    ac自动机匹配，基于词典的命名实体识别
    :param text:
    :return:
    '''
    s = ACMatch(config['ac_test_path'])
    for e in s.ac_match(text):
        print(e)
        logger.info(e)


def test_bilstm_crf_1():
    '''
    一层bilsm_crf命名实体识别
    :param texts:
    :return:
    '''
    blc_cfg = {
        'logger': logger,
        'train_path': config['train_path'],
        'eval_path': config['eval_path'],
        'max_len': 50,
        'batch_size': 128,
        'epoch': 50,
        'loss': 'sgd',
        'rate': 0.01,
        'num_units': 64,
        'num_layers': 1,
        'tf_config': tf_config,
        'model_path': config['bilstm_crf_1_model_path'],
        'summary_path': config['bilstm_crf_1_summary_path'],
        'use_attention': False
    }
    model = BiLstmCrf(**blc_cfg)
    model.fit()
    model.load(config['bilstm_crf_1_predict_path'])
    print(model.predict('招商银行田惠宇行长在股东大会上致辞'))
    model.close()


def test_bilstm_crf_2():
    '''
    两层bilsm_crf命名实体识别
    :param texts:
    :return:
    '''
    blc_cfg = {
        'logger': logger,
        'train_path': config['train_path'],
        'eval_path': config['eval_path'],
        'max_len': 50,
        'batch_size': 128,
        'epoch': 50,
        'loss': 'sgd',
        'rate': 0.01,
        'num_units': 64,
        'num_layers': 1,
        'tf_config': tf_config,
        'model_path': config['bilstm_crf_1_model_path'],
        'summary_path': config['bilstm_crf_1_summary_path'],
        'use_attention': False
    }
    model = BiLstmCrf(**blc_cfg)
    model.fit()
    model.load(config['bilstm_crf_1_predict_path'])
    print(model.predict('招商银行田惠宇行长在股东大会上致辞'))
    model.close()



def test_w2v_bilstm_crf_tf(texts):
    config = loadyaml('conf/NER.yaml')
    logger = setlogger(config)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    w2v = KeyedVectors.load_word2vec_format('./model/w2v/w2v.model', binary=False)
    wblc_cfg = {
        'logger': logger,
        'train_path': './data/ner_train',
        'eval_path': './data/ner_eval',
        'w2v': w2v,
        'max_len': 64,
        'batch_size': 32,
        'epoch': 50,
        'loss': 'sgd',
        'rate': 0.001,
        'num_units': 64,
        'num_layers': 1,
        'tf_config': tf_config,
        'model_path': './model/w2v_bilstm_crf/model',
        'summary_path': './model/w2v_bilstm_crf/summary',
        'use_attention': True
    }
    model = W2VBiLstmCrf(**wblc_cfg)
    model.fit()
    model.load('./model/w2v_bilstm_crf')
    print(model.predict(texts))
    model.close()


def test_bert_crf_tf(text):
    os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4'
    tf_config = tf.ConfigProto(log_device_placement=True)
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # bert配置
    bert_cfg = {
        'bert_path': 'C:/数据/模型/chinese_L-12_H-768_A-12',
        'train_path': './data/train_path',
        'eval_path': './data/eval_path',
        'max_length': 128,
        'batch_size': 32,
        'save_path': './model/bert',
        'learning_rate': 2e-5,
        'epoch': 3,
        'save_checkpoints_steps': 100,
        'tf_config': tf_config
    }
    model = BertCrf(**bert_cfg)
    model.fit()
    print(model.evaluate())
    model.save('./model/bert_crf')
    model = BertCrf(**bert_cfg)
    model.load('./model/bert_crf')
    results = model.predict(text)
    print(results)


def test_bert_bilstm_crf_tf(text):
    # tf配置
    os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4'
    tf_config = tf.ConfigProto(log_device_placement=True)
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # bert配置
    bert_cfg = {
        'bert_path': 'C:/数据/模型/chinese_L-12_H-768_A-12',
        'train_path': './data/ner_train',
        'eval_path': './data/ner_eval',
        'num_units': 128,
        'max_length': 128,
        'batch_size': 32,
        'save_path': '../model/bert',
        'learning_rate': 2e-5,
        'epoch': 3,
        'save_checkpoints_steps': 100,
        'tf_config': tf_config
    }
    model = BertBiLstmCrf(**bert_cfg)
    model.fit()
    print(model.evaluate())
    model.save('./model/bert_bilstm_crf')
    model = BertBiLstmCrf(**bert_cfg)
    model.load('./model/bert_bilstm_crf')
    results = model.predict('招商银行田惠宇行长在股东大会上致辞')
    print(results)


if __name__ == '__main__':
    # test_ac_match('上海天气怎么样')
    test_bilstm_crf_1()