# _*_ coding: utf-8 _*_
import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import vocab
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe

def load_dataset(test_sen=None):

    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied　将句子分成单词列表。 如果sequential = False，沒有ｔｏｋｅｎ被添加
    Field : A class that stores information about the way of preprocessing　存儲processing的方式的信息　https://zhuanlan.zhihu.com/p/31139113
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.   自動pooling到相同長度
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  先將次變成唯一的idx,然後用glove映射到相應的詞嵌入
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.(vocab_size x embedding_dim)的pretrain的東西就產生了
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    ＃將長度差不多的放在一起pading就不用那麼麻煩
    """

    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=35) #寫文本處理的參數設置，可以轉成需要的tensor
    LABEL = data.LabelField(tensor_type=torch.FloatTensor)


    train_data,valid_data,test_data=data.TabularDataset.splits(
        path='./sarcasmdata/ttssvv/rioff/rioffnor/', train='trainData.tsv',test='testData.tsv',validation='devData.tsv', format='tsv',skip_header=True,
        fields=[('text', TEXT), ('label', LABEL)])
    # test_data=data.TabularDataset.splits(
    #     path='./sarcasmdata/ttssvv/',  test='testData.tsv', format='tsv',skip_header=True,
    #     fields=[('text', TEXT), ('label', LABEL)])
    # print(vocab.itos)
    print(train_data[8].__dict__.keys())
    wantdata=test_data[0].text
    print(test_data[0].text)
    # print(vocab.Vocab())
    dimen=300
    TEXT.build_vocab(train_data ,vectors=GloVe(name='6B',dim=dimen))     #Text處理data   min
    LABEL.build_vocab(train_data)
    dict={}
    for id, word in enumerate(TEXT.vocab.itos):
        dict[id]=word
    # print(dict)


    word_embeddings = TEXT.vocab.vectors     # (vocab_size x embedding_dim)的pretrain的東西就產生了
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))
    # print(list(LABEL.vocab))

    # train_data, valid_data = train_data.split() # Further splitting of training_data to create new training_data & validation_data＃將train拆成訓練和驗證
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=8, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)
    #調Ｂａｔｃｈ,排序key,不在不同epoch中重復迭代，不打亂數據
    '''Alternatively we can also use the default configurations下面那個是默認配置'''
    # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)

    vocab_size = len(TEXT.vocab)




    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter,dimen,wantdata,dict
    #TEXT前處理信息, vocab_size單詞表幾個, word_embeddings　input矩陣, train_iter, valid_iter, test_iter #三個集訓練的處理方式：ｂａｔｃｈ ~把相同長度放一起？

