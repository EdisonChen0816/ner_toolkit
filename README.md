# 命名实体识别工具(tensorflow版)

## 词（字）向量模型
采用北师大词（字）向量模型

git：https://github.com/Embedding/Chinese-Word-Vectors

本demo采用的词向量模型： dense + Wikipedia_zh中文维基百科 + Word + 300d

本demo采用的字向量模型： dense + Wikipedia_zh中文维基百科 + Word+Character + 300d

## 分词器：
采用结巴分词器

## 数据：
数据来源：https://www.cluebenchmarks.com/introduce.html  CLUENER细粒度命名实体识别

本demo只识别公司和职位，筛选含有公司和职位的数据，作为训练和验证数据

源数据：/data/raw_data

处理后的训练数据：/data/train.txt

处理后的验证数据：/data/eval.txt

## demo
详见：main.py

ac自动机匹配： test_ac_match

1层bilstm_crf: test_bilstm_crf_1

2层bilstm_crf: test_bilstm_crf_2

3层bilstm_crf: test_bilstm_crf_3

4层bilstm_crf: test_bilstm_crf_4

带attention的bilstm_crf: test_bilstm_crf_attention

## 模型参数及效果
/model中的模型训练参数如下：

**1层bilstm_crf:** 

**2层bilstm_crf:**

**3层bilstm_crf:**

**4层bilstm_crf:**

**带attention的bilstm_crf:**

## todo：
增加带词汇增强的命名实体识别方法

1，FLAT:参考论文《FLAT: Chinese NER Using Flat-Lattice Transformer》