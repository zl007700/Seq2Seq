# coding=utf-8

from tool.config import loadConfig
from model import Seq2SeqModel
from dataset import Seq2SeqDataset

args    = loadConfig('config.ini')
dataset = Seq2SeqDataset(args)
model   = Seq2SeqModel(args)

if args.mode == 'train':
    print('trainging')
    train_set = dataset.getDatas('train')
    eval_set = dataset.getDatas('eval')
    model.train(train_set, eval_set)
elif args.mode == 'eval':
    print('evaluation')
    eval_set = dataset.getDatas('eval')
    model.eval(eval_set)
elif args.mode == 'predict':
    print('prediction')
    eval_set = dataset.getDatas('eval')
    print(dataset.ftk.convert_ids_to_tokens(eval_set[0][0]))
    pred_id = model.predict(eval_set[0][0], eval_set[2][0])
    print(dataset.ftk.convert_ids_to_tokens(pred_id))
elif args.mode == 'freeze':
    print('evaluation')
    model.freeze()
elif args.mode == 'infer':
    print('infer')
    eval_set = dataset.getDatas('eval')
    model.infer(eval_set)
else:
    pass
