# coding=utf-8
"""

Model definition for seq2seq task.

[FUNCTIONS]


"""

import os
import tqdm
import numpy as np
import tensorflow as tf

from blocks.modules import ln, noam_scheme

from tensorflow.layers import Dense
from tensorflow.keras.layers import LSTM, Activation
from tensorflow.contrib import rnn
from tensorflow.contrib.seq2seq import LuongAttention, AttentionWrapper, TrainingHelper
from tensorflow.contrib.seq2seq import BasicDecoder, BeamSearchDecoder, dynamic_decode, tile_batch
from tensorflow.contrib.layers import xavier_initializer

class Seq2SeqModel(object):

    def __init__(self, args):
        self.args = args

        model_op = self.buildModel()
        self.x           = model_op[0]
        self.y           = model_op[1]
        self.x_len       = model_op[2]
        self.y_len       = model_op[3]
        self.logits      = model_op[4]
        self.loss        = model_op[5]
        self.prediction  = model_op[6]
        self.beam_decoder_result_ids = model_op[7]
        self.global_step = model_op[8]
        self.train_op    = model_op[9]
        self.summaries   = model_op[10]

        dir_path = os.path.dirname(__file__)
        self.model_path= os.path.join(dir_path, self.args.model_path)

    def train(self, train_set, eval_set=None):
        ## Restore
        saver = tf.train.Saver()
        sess = tf.Session()
        writer = tf.summary.FileWriter(self.args.log_dir, sess.graph)

        if self.args.restore:
            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
        else:
            print('init all')
            sess.run(tf.global_variables_initializer())

        ## Train
        train_x, train_y, train_x_len, train_y_len = train_set
        sample_num = train_x.shape[0]
        bs = self.args.batch_size

        for epoch_id in range(self.args.epoch):
            for i in tqdm.tqdm(range(sample_num//bs+1)):
                _, loss_val, all_summary, global_step_val = sess.run( [self.train_op, self.loss, self.summaries, self.global_step],
                    feed_dict={
                        self.x: train_x[(i*bs):(i*bs+bs)],
                        self.y: train_y[(i*bs):(i*bs+bs)],
                        self.x_len: train_x_len[(i*bs):(i*bs+bs)],
                        self.y_len: train_y_len[(i*bs):(i*bs+bs)],
                    }
                )
                writer.add_summary(all_summary, global_step_val)

            if epoch_id % self.args.save_period == 0:
                print('loss is %f'%loss_val)
                print('saving model')
                if eval_set != None:
                    self.eval(eval_set, sess)
                save_path = os.path.join(self.model_path, 'model.ckpt')
                saver.save(sess, save_path, global_step = self.global_step, write_meta_graph=False)

        print('saving model')
        save_path = os.path.join(self.model_path, 'model.ckpt')
        saver.save(sess, save_path, global_step = self.global_step, write_meta_graph=False)

        writer.close()
        sess.close()

    def eval(self, test_set, sess=None):
        if not sess:
            sess = tf.Session()
            dir_path = os.path.dirname(__file__)
            self.model_path= os.path.join(dir_path, self.args.model_path)
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))

        eval_x, eval_y, eval_x_len, eval_y_len = test_set
        sample_num = eval_x.shape[0]
        #bs = self.args.batch_size
        bs = 1

        #for i in tqdm.tqdm(range(sample_num//bs+1)):
        for i in tqdm.tqdm(range(10)):
            prediction_val = sess.run(
                self.beam_decoder_result_ids,
                feed_dict={
                    self.x: eval_x[(i*bs):(i*bs+bs)],
                    self.x_len: eval_x_len[(i*bs):(i*bs+bs)],
                }
            )
            ground_truth = eval_y[(i*bs):(i*bs+bs)].tolist()

            print('GROUND TRUE', ground_truth[0])
            print('PREDICTION', prediction_val)


    def freeze(self):
        from tensorflow.python.framework import graph_util

        dir_path = os.path.dirname(__file__)
        self.model_path= os.path.join(dir_path, self.args.model_path)
        pb_file = os.path.join(self.model_path, 'model.pb')

        logits = tf.identity(self.logits, 'logits')
        prediction = tf.identity(self.prediction, 'prediction')

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['logits', 'prediction'])

        with tf.gfile.GFile(pb_file, 'wb') as f:
            f.write(constant_graph.SerializeToString())

        print('Freezing Done')

    def infer(self, infer_set):
        dir_path = os.path.dirname(__file__)
        self.model_path= os.path.join(dir_path, self.args.model_path)
        pb_file = os.path.join(self.model_path, 'model.pb')

        with tf.gfile.FastGFile(pb_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            #all_tensor = [n.name for n in graph_def.node]
            #for t in all_tensor:
            #    print(t)

            logits, prediction = tf.import_graph_def(graph_def,return_elements=['logits:0', 'prediction:0'])

        sess = tf.Session()
        x = sess.graph.get_tensor_by_name('import/input/inputs:0')
        x_len = sess.graph.get_tensor_by_name('import/input/inputs_len:0')

        eval_x, eval_y, eval_x_len = infer_set
        sample_num = eval_x.shape[0]
        bs = self.args.batch_size

        eval_all = [0,0]
        eval_dict = [[0,0,0] for i in range(self.args.num_labels)]

        for i in tqdm.tqdm(range(sample_num//bs)):
            prediction_val = sess.run(
                prediction,
                feed_dict={
                    x: eval_x[(i*bs):(i*bs+bs)],
                    x_len: eval_x_len[(i*bs):(i*bs+bs)],
                }
            )
            ground_truth = eval_y[(i*bs):(i*bs+bs)].tolist()

            for j in range(bs):
                pred = prediction_val[j]
                true = ground_truth[j]
                eval_dict[pred][1] += 1
                eval_dict[true][2] += 1
                eval_all[0] += 1
                if pred == true:
                    eval_dict[pred][0] += 1
                    eval_all[1] += 1
        print('Accuracy is : ', eval_all[1] / eval_all[0])

        for label_id, eval_result in enumerate(eval_dict):
            if eval_result[1] == 0:
                eval_result[1] = 2e32
            if eval_result[2] == 0:
                eval_result[2] = 2e32
            print('Accuracy && Recall of label %d is : %f %f'%(label_id, eval_result[0]/eval_result[1], eval_result[0]/eval_result[2]))

    def buildModel(self):
        T_in  = self.args.T_in
        T_out = self.args.T_out 
        D_in  = self.args.D_in
        D_out = self.args.D_out
        E     = self.args.embedding_dim
        H     = self.args.hidden_dim
        SOS   = self.args.SOS
        EOS   = self.args.EOS
        PAD   = self.args.PAD
        beam_width = 3

        # Input
        with tf.name_scope('input'):
            x = tf.placeholder(shape=(None, T_in), dtype=tf.int32, name='encoder_inputs')
            # N, T_out
            y = tf.placeholder(shape=(None, T_out), dtype=tf.int32, name='decoder_inputs')
            # N
            x_len = tf.placeholder(shape=(None,), dtype=tf.int32)
            # N
            y_len = tf.placeholder(shape=(None,), dtype=tf.int32)
            # dynamic sample num
            batch_size = tf.shape(x)[0]
        
            # symbol mask
            sos = tf.ones(shape=(batch_size, 1), dtype=tf.int32) * SOS
            eos = tf.ones(shape=(batch_size, 1), dtype=tf.int32) * EOS
            pad = tf.ones(shape=(batch_size, 1), dtype=tf.int32) * PAD

            # input mask
            x_mask = tf.sequence_mask(x_len, T_in, dtype=tf.float32)
            y_with_sos_mask = tf.sequence_mask(y_len, T_out+1, dtype=tf.float32)
            y_with_pad = tf.concat([y, pad], axis=1)
            eos_mask = tf.one_hot(y_len, depth = T_out+1, dtype=tf.int32) * EOS
        
            # masked inputs
            y_with_eos = y_with_pad + eos_mask
            y_with_sos = tf.concat([sos, y], axis=1)

        ## Embedding
        with tf.name_scope('embedding'):
            if self.args.use_pretrained:
                embedding_pretrained = np.fromfile(self.args.pretrained_file, dtype=np.float32).reshape((-1, E))
                embedding = tf.Variable(embedding_pretrained, trainable=False)
            else:
                embedding = tf.get_variable(name='embedding', shape=(D_in, E), dtype=tf.float32, initializer=xavier_initializer())
            e_x = tf.nn.embedding_lookup(embedding, x)
            e_y = tf.nn.embedding_lookup(embedding, y_with_sos)
            if self.args.mode == 'train':
                e_x = tf.nn.dropout(e_x, self.args.keep_prob)

        ## Encoder
        with tf.name_scope('encoder'):
            ## BiLSTM
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(H, forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(H, forget_bias=1.0, state_is_tuple=True)
            (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, 
                                                    lstm_bw_cell, 
                                                    e_x,
                                                    sequence_length=x_len,
                                                    dtype=tf.float32,
                                                    time_major=False,
                                                    scope=None)
            encoder_output = output_fw  + output_bw
            encoder_final_state = state_fw[0] + state_bw[0]

        ## Decoder
        with tf.name_scope('decoder'):
            decoder_cell = rnn.GRUCell(num_units = H)
            decoder_lengths = tf.ones(shape=[batch_size], dtype=tf.int32) * (T_out + 1)

            ## Trainning decoder
            with tf.variable_scope('attention'):
                attention_mechanism = LuongAttention(num_units=H,
                                                     memory=encoder_output,
                                                     memory_sequence_length=x_len,
                                                     name = 'attention_fn')
            projection_layer = Dense(units = D_out, kernel_initializer = xavier_initializer())

            train_decoder_cell = AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism, attention_layer_size=None)
            train_decoder_init_state = train_decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_final_state)
            training_helper = TrainingHelper(e_y, decoder_lengths, time_major=False)
            train_decoder = BasicDecoder(cell=train_decoder_cell,
                                                    helper=training_helper,
                                                    initial_state=train_decoder_init_state,
                                                    output_layer=projection_layer)
            train_decoder_outputs, _, _ = dynamic_decode(train_decoder,
                                                    impute_finished=True,
                                                    maximum_iterations=T_out+1)
            # N, T_out+1, D_out
            train_decoder_outputs = ln(train_decoder_outputs.rnn_output)
        
            ## Beam_search decoder
            beam_memory        = tile_batch(encoder_output, beam_width)
            beam_memory_state  = tile_batch(encoder_final_state, beam_width)
            beam_memory_length = tile_batch(x_len, beam_width)

            with tf.variable_scope('attention', reuse=True):
                beam_attention_mechanism = LuongAttention(num_units=H,
                                                     memory=beam_memory,
                                                     memory_sequence_length=beam_memory_length,
                                                     name = 'attention_fn')
            beam_decoder_cell = AttentionWrapper(cell=decoder_cell,
                                                     attention_mechanism=beam_attention_mechanism,
                                                     attention_layer_size=None)
            beam_decoder_init_state = beam_decoder_cell.zero_state(batch_size=batch_size*beam_width,
                                                     dtype=tf.float32).clone(cell_state=beam_memory_state)
            start_tokens = tf.ones((batch_size), dtype=tf.int32)  * SOS
            beam_decoder = BeamSearchDecoder(cell=beam_decoder_cell,
                                                     embedding=embedding,
                                                     start_tokens=start_tokens,
                                                     end_token=EOS,
                                                     initial_state=beam_decoder_init_state,
                                                     beam_width=beam_width,
                                                     output_layer=projection_layer)
            beam_decoder_outputs, _, _ = dynamic_decode(beam_decoder,
                                                     scope=tf.get_variable_scope(),
                                                     maximum_iterations=T_out+1)
            beam_decoder_result_ids = beam_decoder_outputs.predicted_ids
        
        with tf.name_scope('loss'):
            logits = tf.nn.softmax(train_decoder_outputs)
            cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_with_eos, logits)
            loss_mask = tf.sequence_mask(y_len+1, T_out+1, dtype=tf.float32)
            loss = tf.reduce_sum(cross_entropy*loss_mask) / tf.cast(batch_size, dtype=tf.float32)
            prediction = tf.argmax(logits, 2)

        ## train_op
        with tf.name_scope('train'):
            global_step = tf.train.get_or_create_global_step()
            lr = noam_scheme(self.args.lr, global_step, self.args.warmup_steps)
            optimizer = tf.train.AdamOptimizer(lr)
        
            ## gradient clips
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.args.gradient_clip_num)
            train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params), global_step=global_step)
        
        # Summary
        with tf.name_scope('summary'):
            tf.summary.scalar('lr', lr)
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('global_step', global_step)
            summaries = tf.summary.merge_all()
        return x, y, x_len, y_len, logits, loss, prediction, beam_decoder_result_ids, global_step, train_op, summaries
