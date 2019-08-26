import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable
class CoattentionNet(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights,heat):
    # def __init__(self, vocab_size, embedding_dim):  # =1000?
        super().__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.heat = heat

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)       #, padding_idx=0    ????
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)



        self.unigram = nn.Conv1d(embedding_length, embedding_length, 1)     #input output
        self.bigram = nn.Conv1d(embedding_length, embedding_length, 2)
        self.trigram = nn.Conv1d(embedding_length, embedding_length, 3)
        self.maxp = nn.MaxPool2d(kernel_size=(3, 1))  # 取橫向最大
        self.word_parallel = ParallelCoattention(D=embedding_length)
        self.phrase_parallel = ParallelCoattention(D=embedding_length)
        self.dp = nn.Dropout(p=0.5)
        self.tanh = nn.Tanh()
        self.label = nn.Linear(embedding_length, output_size)
        # self.lstm = nn.LSTM(embedding_length, 300)

        self.pe = PositionalEncoder(embedding_length)






    def forward(self, input_sentence, batch_size=None):
        word_embeddings = self.word_embeddings(input_sentence)    # 32 ,20,300
        # word_embeddings=self.pe(word_embeddings)


        word_embeddings = word_embeddings.transpose(2, 1) #[32, 300, 20
        unigram = self.tanh(self.unigram(word_embeddings))#[32, 300, 20
        bigram = self.tanh(self.bigram(
            torch.cat((torch.zeros(word_embeddings.shape[0], word_embeddings.shape[1], 1).cuda(),word_embeddings),
            dim=2)))
        # [32, 300, 20
        trigram = self.tanh(self.trigram(   #
            torch.cat(( torch.zeros(word_embeddings.shape[0], word_embeddings.shape[1], 1).cuda(),word_embeddings, torch.zeros(word_embeddings.shape[0], word_embeddings.shape[1], 1).cuda()),         #我覺得應該前面補0
                      dim=2)))
        # print('bigram', bigram.size())# All-Grams [bx512xT]       可print size
        kilogram = torch.cat((                                                  #32, 300, 3, 20]
            unigram.view(unigram.shape[0], unigram.shape[1], 1, unigram.shape[2]),
            bigram.view(bigram.shape[0], bigram.shape[1], 1, bigram.shape[2]),
            trigram.view(trigram.shape[0], trigram.shape[1], 1, trigram.shape[2])
        ), dim=2)
        heatm=self.heat
        kilogram = self.maxp(kilogram)          #32, 300, 1, 20
        kilogram = kilogram.view(kilogram.shape[0], kilogram.shape[1], kilogram.shape[3])  #32,300,20

        # kilogram = self.pe(kilogram.transpose(2, 1))
        # kilogram=kilogram.transpose(2,1)

        # f_w1 = self.word_parallel(word_embeddings,heatm)   #32x25x1         #目前還沒有加入全部進去的結果,,,,,,,,,,,,,,,,,,,,自己和自己不能算?
        # f_w=torch.bmm(f_w1.transpose(2, 1), word_embeddings.transpose(2, 1))        #32x1x25   ,32x25x300     #可以考慮不要weighted sum
        # f_w=torch.squeeze(f_w, 1)   #32x300


        f_p1=self.phrase_parallel(kilogram,heatm)       #32x25x1
        # f_pp=torch.mul(f_p1.transpose(2, 1), kilogram)      #
        # output, (final_hidden_state, final_cell_state) = self.lstm(f_pp.permute(2, 0, 1))
        # final_output = self.label(final_hidden_state[-1])
        f_p = torch.bmm(f_p1.transpose(2, 1), kilogram.transpose(2, 1))                #可以考慮不要weighted sum
        f_p = torch.squeeze(f_p, 1)

        # all=torch.cat((unigram,bigram,trigram),dim=2)
        # f_con1=self.word_parallel(all,heatm)
        # f_con=torch.bmm(f_con1.transpose(2, 1), all.transpose(2, 1))
        # f_con = torch.squeeze(f_con, 1)

        final=self.label(f_p)  #300x2    32x300   #32x2
        return final,torch.squeeze(f_p1,2)






class ParallelCoattention(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.W_b_weight = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(D, D)))
        self.tanh = nn.Tanh()
        self.dp = nn.Dropout(p=0.5)
        self.maxp = nn.MaxPool1d(kernel_size=D)  #2d?
    def forward(self, embedding,heatm):
        C = self.dp(self.tanh(torch.matmul(torch.matmul(embedding.transpose(2, 1), self.W_b_weight), embedding)))


        # print('C size:',C.size())
        # EMBEDING=DXT #C=TXT
        Max_cow=self.maxp(C)
        # print('Max_cow size:', Max_cow.size())
        a_q = F.softmax(Max_cow, dim=1)  #32x1x35

        # if heatm:
        #
        #     print(torch.squeeze(a_q,2).cpu().numpy().tolist())
        b = torch.bmm(a_q.transpose(2, 1), embedding.transpose(2, 1))       #aq=32x1x35           embedding=32, 300, 35
        # print('b size:',b.size())               #32x1x300
        return a_q          #




class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=35):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], \
                         requires_grad=False).cuda()
        return x





