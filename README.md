# -基于层级attention机制的多标签预测-
对裁判文书内容中的罪犯进行罪名提取，把罪名预测转化为标签预测，进行以一人多罪、数罪并罚的多罪名预测任务。模型用层级attention机制作为编码器对裁判文书的犯罪事实描述内容进行编码，用LSTM作为解码器基本单元对信息表征进行罪名预测。模型一方面将标签之间的逻辑关系融入到模型中, 另一方面加强了编码器与解码器之间的信息流动。模型通过在CAIL，CJO 等多标签罪名数据集上的模型训练及使用并对比包括BR,CC,LP,Fact_law,HAN+threshold,RCNN,TextCNN,Seq2Seq+att等多标签预测模型，我们的模型在两个数据集上都取得了最优的结果。

In order to extract the charge of the criminal in the judgment document description, we transform the multi-charge prediction into the multi-label prediction and carry out the task of multi-charge prediction of one person with multiple crimes and multiple punishments. In our proposed model, the nested attention mechanism is used as the encoder to encode the criminal fact description, and the LSTM is used as the basic unit of the decoder to predict the accusation of the information representation. On the one hand, the model integrates the logical relationship between charges into the model, on the other hand, it enhances the information flow through the encoder-decoder attention. The model is used in multi-charge datasets CAIL and CJO, and compared with BR, CC, LP and fact_ Law, Han + threshold, RCNN, textcnn, seq2seq + att and other multi label prediction models, our model has achieved the best results on both datasets.

# -Paper data and code-
This is the code for the IEEE ACCESS Paper: [Sequence Generation Network Based on Hierarchical Attention for Multi-Charge Prediction](https://ieeexplore.ieee.org/abstract/document/9103518). We have implemented our methods in both Tensorflow.
Here are two datasets we used in our paper：

CJO :http://wenshu.court.gov.cn/

CAIL:http://cail.cipsc.org.cn/index.html

# -Usage-
You need to run the file `datasets/preprocess.py` first to preprocess the data

For example: `cd datasets; python preprocess.py --dataset=sample`

Then you can run the file `tensorflow_code/main.py` to train the model.

For example: cd tensorflow_code; python main.py --dataset=sample

You can also change other parameters according to the usage:

`usage: main.py [-h] [--dataset DATASET] [--batchSize BATCHSIZE]`
`               [--hiddenSize HIDDENSIZE] [--epoch EPOCH] [--lr LR]`
`               [--lr_dc LR_DC] [--lr_dc_step LR_DC_STEP] [--l2 L2]`
`               [--step STEP] [--patience PATIENCE] [--nonhybrid]`
`              [--validation] [--valid_portion VALID_PORTION]`

`optional arguments:`
`  -h, --help            show this help message and exit`
`  --dataset DATASET     dataset name:`
`                        diginetica/yoochoose1_4/yoochoose1_64/sample`
`  --batchSize BATCHSIZE`
                        input batch size
  --hiddenSize HIDDENSIZE
                        hidden state size
  --epoch EPOCH         the number of epochs to train for
  --lr LR               learning rate
  --lr_dc LR_DC         learning rate decay rate
  --lr_dc_step LR_DC_STEP
                        the number of epochs after which the learning rate
                        decay
  --l2 L2               l2 penalty
  --step STEP           gnn propogation steps
  --patience PATIENCE   the number of epoch to wait before early stop
  --nonhybrid           only use the global preference to predict
  --validation          validation
  --valid_portion VALID_PORTION
                        split the portion of training set as validation set`
