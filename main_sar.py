import os
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from models.co_attention import CoattentionNet

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
from pandas.core.frame import DataFrame

TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter,dimen,wantdata,dict = load_data.load_dataset()    #reture 的東西抓出來用

def clip_gradient(model, clip_value):           #可能就是裁剪梯度吧
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train_iter, epoch):          #訓練過程ｉｎｐｕｔ 模型：ＬＳＴＭ　
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=1e-3, weight_decay=1e-5)      #改過adam
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not 8):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction,atw = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        # clip_gradient (model, 1e-1)#前面define　
        optim.step()
        steps += 1
        
        if steps % 100 == 0:
            print ('Epoch: {0:02}'.format(epoch+1), 'Idx: {0:02}'.format(idx+1), 'Training Loss: {0:.4f}'.format(loss.item()), 'Training Accuracy: {0: .2f}%'.format(acc.item()))
            # print('Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)


def calculate_metric(metric_fn, true_y, pred_y):
    # multi class problems need to have averaging method
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)
def print_scores(p, r, f1, a, batch_size):
    # just an utility printing function
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print('\t{0}'.format(name) ,  '{0:.4f}'.format(sum(scores)/batch_size))


def eval_model(model, val_iter,jieguo=False):
    total_epoch_loss = 0

    model.eval()
    precision, recall, f1, accuracy = [], [], [], []
    ppp=[]
    lll=[]
    ttt=[]
    www=[]

    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            # print(text)
            if (text.size()[0] is not 8):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()          #Y


            prediction,atw = model(text)   #outputs      atw =weight

            loss = loss_fn(prediction, target)
            total_epoch_loss += loss.item()  # val_losses
            predicted_classes = torch.max(prediction, 1)[1]
            # if jieguo == True:
                # print('text',text.cpu().numpy().tolist())
                # print('label:',target.cpu().numpy().tolist())
                # print('predict:',predicted_classes.cpu().numpy().tolist())
                # print('weight',atw.cpu().tolist())


            ttt=ttt+text.cpu().numpy().tolist()

            lll=lll+target.cpu().numpy().tolist()
            ppp=ppp+(predicted_classes.cpu().numpy().tolist())
            www=www+(atw.cpu().tolist())


            for acc, metric in zip((precision, recall, f1, accuracy),(precision_score, recall_score, f1_score, accuracy_score)):
                acc.append( calculate_metric(metric, target.cpu(), predicted_classes.cpu()) )

    if jieguo == True:
        # zero=[0]
        ttt = [[dict[j] for j in i] for i in ttt]
        # ttt=[i+list2bigram(zero+i)+list2trigram(zero+i+zero)for i in ttt]       #gram才這樣
        save={"textword":ttt,
              "labels": lll,
              "predict":ppp,
              "weight":www}
        data = DataFrame(save)
        # data.to_csv('bijiao/ptackmax.txt',sep='\t', index=False)

        # print(len(lll))
        # print(len(ppp))
        # print(len(www))
        # ttt=[[dict[j] for j in i] for i in ttt]
        # print(len(ttt))
        # for j in range(480):
        #     for i in ttt,lll,ppp,www:
        #
        #         f.write(str(i[j]))
        #         f.write('\n')
        # f1=open('bijiao/rioff.txt','r+')
        # new=open('bijiao/rioffword.txt','w+')
        # count=4
        # cun=[]
        # for i in f1:
        #     count=count+1
        #     cun.append(i)
        #     new.write(i.strip())
        #     new.write('\t')
        #     if count%4==0:
        #         new.write('\n')

    print(' final loss:{0:.5f}'.format(total_epoch_loss/len(val_iter)))

    print_scores(precision, recall, f1, accuracy, len(val_iter))
	

learning_rate = 1e-2
batch_size = 8
output_size = 2
hidden_size =512
embedding_length = dimen

model=CoattentionNet(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings,heat=False)

loss_fn = F.cross_entropy

for epoch in range(10):
    train_loss, train_acc = train_model(model, train_iter, epoch)
    print('Epoch: {0:02}'.format(epoch + 1), 'Train Loss: {0:.3f}'.format(train_loss),
          'Train Acc: {0:.2f}%'.format(train_acc))
    # val_loss, val_acc = eval_model(model, valid_iter)                                               #加recall f1 precision
    if epoch==699:
        print("*********")
        model.heat = True
        eval_model(model, valid_iter,jieguo=True)

    eval_model(model, valid_iter)

from flask import Flask, request, send_from_directory, render_template, redirect
import flask, flask.views
app = Flask('test_app')
app.debug = True

@app.route('/index')
def index():
	print('in function index')
	return flask.render_template('index.html')
@app.route('/about')
def about():
	print('in function about')
	# return redirect(url_for("about"))
	return flask.render_template('about.html')
@app.route('/contact')
def contact():
	print('in function contact')
	return render_template('contact.html')
@app.route('/submit', methods=['POST'])
def submit():
    print('in function submit')
    print(request.form['haha'])
    text = request.form['haha']
    text = TEXT.preprocess(text)
    text = [[TEXT.vocab.stoi[x] for x in text]]
    text = np.asarray(text)
    text = torch.LongTensor(text)
    text = Variable(text, volatile=True)
    text = text.cuda()
    model.eval()
    output, atw = model(text, 1)
    out = F.softmax(output, 1)
    if (torch.argmax(out[0]) == 1):
        print("sarcasm")
        return flask.render_template('submit.html', text="This is a sarcasm text")
    else:
        print("Nonsarcasm")
        return flask.render_template('submit.html', text="This is a non-sarcasm text")
if(__name__ == '__main__'):
	app.run(host='127.0.0.1')
# test_sen1 = "I  hate work"
# test_sen1 = TEXT.preprocess(test_sen1)
# test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]
# test_sen = np.asarray(test_sen1)
# test_sen = torch.LongTensor(test_sen)
# test_tensor = Variable(test_sen, volatile=True)
# test_tensor = test_tensor.cuda()
# model.eval()
# output,atw = model(test_tensor, 1)
# print(atw)
# out = F.softmax(output, 1)
# if (torch.argmax(out[0]) == 1):
#     print ("sarcasm")
# else:
#     print ("Nonsarcasm")






# model.heat = True
# eval_model(model, test_iter)        #test
# print('Test Loss: {0:.3f}'.format(test_loss), 'Test Acc: {0:.2f}%'.format(test_acc))

# ''' Let us now predict the sentiment on a single sentence just for the testing purpose. '''
#
#
# test_sen1 = "I love bitch"
#
#
# test_sen1 = TEXT.preprocess(test_sen1)
# print(test_sen1)
# test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]
# print(test_sen1)
# # test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."
# # test_sen2 = TEXT.preprocess(test_sen2)
# # test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]
#
# test_sen = np.asarray(test_sen1)
# test_sen = torch.LongTensor(test_sen)
# test_tensor = Variable(test_sen, volatile=True)
# test_tensor = test_tensor.cuda()
# print(test_tensor)
# model.heat = False
# model.eval()
#
# output = model(test_tensor, 1)
#
# out = F.softmax(output, 1)
# if (torch.argmax(out[0]) == 1):
#     print ("Sentiment: Sarc")
# else:
#     print ("Sentiment: Not Sarc")






