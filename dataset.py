#coding=utf-8 
"""
[INPUT]
    ./data/corpus.txt
        -- This is default input file. Please put your corpus file in this directory and preprocess it into supporting structure.
    Supporting input file structure:
        corpus.txt >>
        question1\tanswer1
        question2\tanswer2
        question3\tanswer3
                .
                .
[OUTPUT]
    ./build/train_x.txt
    ./build/eval.txt
        -- These files are automatically splited by program from corpus.txt.

    ./build/train_x.bin
    ./build/train_x_len.bin
    ./build/train_y.bin
    ./build/train_y_len.bin
        -- These binary files are processed numpy structure, which can be directly used by model.
"""

import os
import sys
import random
import numpy as np
from blocks.tokenization import FullTokenizer

class Seq2SeqDataset(object):

    def __init__(self, args):
        self.args = args
        self.ftk = FullTokenizer(self.args.vocab_file, chinese_word_seg=True)
        dir_path = os.path.dirname(__file__)
        self.corpus_file_path = os.path.join(dir_path, self.args.data_path, 'corpus.txt') 
        self.train_file_path  = os.path.join(dir_path, self.args.build_path, 'train.txt') 
        self.eval_file_path   = os.path.join(dir_path, self.args.build_path, 'eval.txt') 

        self.splitCorpus()
        self.preproces(self.train_file_path, 'train')
        self.preproces(self.eval_file_path, 'eval')

    def splitCorpus(self):
        if not os.path.exists(self.args.build_path):
            os.mkdir(self.args.build_path)

        if not os.path.exists(self.train_file_path) or not os.path.exists(self.eval_file_path):
            print('Split corpus.')
            datas = []
            with open(self.corpus_file_path, 'r') as fin:
                for line in fin:
                    datas.append(line)
            train_num = int(len(datas) * 0.8)
            eval_num = len(datas) - train_num

            random.shuffle(datas)

            with open(self.train_file_path, 'w') as fout:
                for data in datas[:train_num]:
                    fout.write(data)
            with open(self.eval_file_path, 'w') as fout:
                for data in datas[train_num:]:
                    fout.write(data)

    def preproces(self, data_file, file_tag):
        output_x_file     = os.path.join(self.args.build_path,'%s_x.bin'%file_tag)
        output_y_file     = os.path.join(self.args.build_path,'%s_y.bin'%file_tag)
        output_x_len_file = os.path.join(self.args.build_path,'%s_x_len.bin'%file_tag)
        output_y_len_file = os.path.join(self.args.build_path,'%s_y_len.bin'%file_tag)

        if os.path.exists(output_x_file) and os.path.exists(output_y_file) and os.path.exists(output_x_len_file) and os.path.exists(output_y_len_file):
            return

        print('Processing ', file_tag)

        texts_A, texts_B = [],[]
        texts_A_len, texts_B_len = [],[]
        with open(data_file, 'r', encoding='utf-8') as fin:
            for line_id, line in enumerate(fin):
                try:
                    text_A, text_B = line.split('\t')

                    tokens_A = self.ftk.tokenize(text_A)
                    tokens_B = self.ftk.tokenize(text_B)

                    texts_A.append(self.ftk.convert_tokens_to_ids_with_padding(tokens_A, self.args.T_in))
                    texts_B.append(self.ftk.convert_tokens_to_ids_with_padding(tokens_B, self.args.T_out))
                    texts_A_len.append(self._truncate(len(tokens_A), self.args.T_in))
                    texts_B_len.append(self._truncate(len(tokens_B), self.args.T_out))
                except Exception as e:
                    print(e)
                if line_id % 10000 == 0 and line_id!=0:
                    print(line_id, 'done')

        data_x = np.array(texts_A).astype(np.int32)
        data_y = np.array(texts_B).astype(np.int32)
        data_x_len = np.array(texts_A_len).astype(np.int32)
        data_y_len = np.array(texts_B_len).astype(np.int32)
        
        print(data_x[:2])
        print(data_x_len[:2])
        print(data_y[:2])
        print(self.ftk.convert_ids_to_tokens(data_x[0]))

        data_x.tofile(output_x_file)
        data_y.tofile(output_y_file)
        data_x_len.tofile(output_x_len_file)
        data_y_len.tofile(output_y_len_file)

    def getDatas(self, tag='train'):
        x_file     = os.path.join(self.args.build_path,'%s_x.bin'%tag)
        y_file     = os.path.join(self.args.build_path,'%s_y.bin'%tag)
        x_len_file = os.path.join(self.args.build_path,'%s_x_len.bin'%tag)
        y_len_file = os.path.join(self.args.build_path,'%s_y_len.bin'%tag)

        data_x     = np.fromfile(x_file, dtype=np.int32).reshape((-1, self.args.T_in))
        data_y     = np.fromfile(y_file, dtype=np.int32).reshape((-1, self.args.T_out))
        data_x_len = np.fromfile(x_len_file, dtype=np.int32)
        data_y_len = np.fromfile(y_len_file, dtype=np.int32)
        return data_x, data_y, data_x_len, data_y_len

    def _truncate(self, input_size, cut_off_size):
        if input_size > cut_off_size:
            return cut_off_size
        else:
            return input_size
