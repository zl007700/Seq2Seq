# coding=utf-8

from tool.config import loadConfig
from model import Seq2SeqModel
from dataset import Seq2SeqDataset

args    = loadConfig('config.ini')
model   = Seq2SeqModel(args)
dataset = Seq2SeqDataset(args)

if args.mode == 'train':
    print('trainging')
    train_set = dataset.getDatas('train')
    model.train(train_set)
elif args.mode == 'eval':
    print('evaluation')
    eval_set = dataset.getDatas('eval')
    model.eval(eval_set)
elif args.mode == 'freeze':
    print('evaluation')
    model.freeze()
elif args.mode == 'infer':
    print('infer')
    eval_set = dataset.getDatas('eval')
    model.infer(eval_set)
else:
    pass
