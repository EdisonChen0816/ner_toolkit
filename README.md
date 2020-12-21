# 命名实体识别工具(tensorflow版)


## 字向量模型
我们自己训练的模型，采用中文维基百科数据作为训练语料

模型存放在百度云中

链接：https://pan.baidu.com/s/1C1qB2b6HyzOpj3eqDehhEQ 提取码：5b6y

下载w2v.model.vectors.npy，放在 /model/w2v 目录下


## 数据：
ac自动机测试数据是省市区数据，来自网络。

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

bilstm_crf + attention: test_bilstm_crf_attention

w2v + 1层bilstm_crf: test_w2v_bilstm_crf_1

w2v + 2层bilstm_crf: test_w2v_bilstm_crf_2

w2v + 1层bilstm_crf + attention: test_w2v_bilstm_crf_attention

bert_crf: test_bert_crf

bert + bilstm_crf: test_bert_bilstm_crf


## 模型参数及效果
/model中的模型训练参数如下：

**(1) 1层bilstm_crf:** 

batch_size: 64, epoch: 200, loss: adam, rate: 0.01, num_unit: 256, dropout: 0.1

eval company recall:0.7301587301587301, eval company precision:0.7582417582417582, eval company f1:0.7439353099730458

eval position recall:0.7505773672055427, eval position precision:0.818639798488665, eval position f1:0.7831325301204818


**(2) 2层bilstm_crf:**

batch_size: 64, epoch: 200, loss: adam, rate: 0.01, num_unit: 220, dropout: 0.0

eval company recall:0.6984126984126984, eval company precision:0.7213114754098361, eval company f1:0.7096774193548387

eval position recall:0.7251732101616628, eval position precision:0.7929292929292929, eval position f1:0.7575392038600723


**(3) 3层bilstm_crf:**

batch_size: 128, epoch: 200, loss: adam, rate: 0.01, num_unit: 200, dropout: 0.1

eval company recall:0.6851851851851852, eval company precision:0.7442528735632183, eval company f1:0.7134986225895316

eval position recall:0.7090069284064665, eval position precision:0.7752525252525253, eval position f1:0.7406513872135103


**(4) 4层bilstm_crf:**

batch_size: 128, epoch: 200, loss: adam, rate: 0.01, num_unit: 150, dropout: 0.0

eval company recall:0.6772486772486772, eval company precision:0.7272727272727273, eval company f1:0.7013698630136986

eval position recall:0.7251732101616628, eval position precision:0.7969543147208121, eval position f1:0.7593712212817412


**(5) 带attention的bilstm_crf:**

batch_size: 64, epoch: 200, loss: adam, rate: 0.01, num_unit: 200, dropout: 0.1

eval company recall:0.7037037037037037, eval company precision:0.7327823691460055, eval company f1:0.717948717948718

eval position recall:0.7482678983833718, eval position precision:0.8120300751879699, eval position f1:0.7788461538461539


**(6) w2v + 一层bilstm_crf:**

batch_size: 64, epoch: 200, loss: adam, rate: 0.01, num_unit: 200, dropout: 0.1

eval company recall:0.7433862433862434, eval company precision:0.7762430939226519, eval company f1:0.7594594594594595

eval position recall:0.7829099307159353, eval position precision:0.8411910669975186, eval position f1:0.8110047846889952


**(7) w2v + 两层bilstm_crf:**

batch_size: 64, epoch: 200, loss: adam, rate: 0.01, num_unit: 200, dropout: 0.1

eval company recall:0.7354497354497355, eval company precision:0.7616438356164383, eval company f1:0.7483176312247644

eval position recall:0.792147806004619, eval position precision:0.8051643192488263, eval position f1:0.7986030267753202


**(8) w2v + bilstm_crf + attention:**

batch_size: 64, epoch: 200, loss: adam, rate: 0.01, num_unit: 200, dropout: 0.1

eval company recall:0.7328042328042328, eval company precision:0.7824858757062146, eval company f1:0.7568306010928961

eval position recall:0.74364896073903, eval position precision:0.817258883248731, eval position f1:0.7787182587666264


**(9) bert_crf:**


batch_size: 32, epoch: 500, loss: sgd, rate: 0.01, max_len: 64, encoder_layer: 11

eval company recall:0.6613756613756614, eval company precision:0.7246376811594203, eval company f1:0.6915629322268325

eval position recall:0.7222222222222222, eval position precision:0.78, eval position f1:0.7500000000000001


**(10) bert + bilstm_crf:**

batch_size: 32, epoch: 500, loss: sgd, rate: 0.01, num_units: 128, dropout: 0.1, max_len: 64, encoder_layer: 11

eval company recall:0.7435897789091218, eval company precision:0.7846376811594203, eval company f1:0.7635624605658261

eval position recall:0.7832222222222222, eval position precision:0.8446376811594203, eval position f1:0.812771418764053


## todo：
增加带词汇增强的命名实体识别方法

1，FLAT:参考论文《FLAT: Chinese NER Using Flat-Lattice Transformer》