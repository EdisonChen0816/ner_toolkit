# 命名实体识别工具

## 词向量模型
下载w2v.model.vectors.npy，放在 model/w2v 目录下

链接：https://pan.baidu.com/s/1C1qB2b6HyzOpj3eqDehhEQ 提取码：5b6y

# demo
## ac自动机匹配
main.py test_ac_match

## tensorflow版 bilstm_crf
main.py test_bilstm_crf_tf

## tensorflow版 w2v + bilstm_crf
main.py test_w2v_bilstm_crf_tf

## tensorflow版 bert_crf
main.py test_bert_crf_tf

## tensorflow版 bert_bilstm_crf
main.py test_bert_bilstm_crf

# todo：
## 带词汇增强的命名实体识别方法
1，FLAT:参考论文《FLAT: Chinese NER Using Flat-Lattice Transformer》