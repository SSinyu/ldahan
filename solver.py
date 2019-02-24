import os
import numpy as np
import tensorflow as tf
from networks import LDA2vec_HAN


class Solver(object):
    def __init__(self, args, train_loader=None, test_loader=None):
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.save_path = args.save_path
        self.min_day = args.min_day
        self.max_day = args.max_day
        self.n_user = len(train_loader.log_seq)

        self.vocab_size = len(train_loader.log_vocab)
        self.n_topics = args.n_topics
        self.window_size = args.window_size
        self.target_hour = args.target_hour
        self.batch_size = args.batch_size

        if args.pre_embed:
            self.embed_init_vec = np.load(os.path.join(args.data_path, 'embed.npy'))
            self.embed_size = self.embed_init_vec.shape[1]
        else:
            self.embed_size = args.embed_size
        self.embed_tuning = args.embed_tuning
        # hidden_size = embed_size/2

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.eval_iters = args.eval_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters

        self.lr = args.lr
        self.clip = args.clip

        self.LDA_HAN = LDA2vec_HAN(self.n_user, self.vocab_size, self.embed_size, self.n_topics, pre_embed=args.pre_embed, embed_tuning=self.embed_tuning)
        self.saver = tf.train.Saver(tf.global_variables())
        self.saver.export_meta_graph(os.path.join(self.save_path, 'LDA_HAN.meta'))

        # gradient clipping
        self.global_step = tf.Variable(0, trainable=False)
        optim = tf.train.AdamOptimizer(self.lr)
        tv_ = tf.trainable_variables()
        gradient_, _ = tf.clip_by_global_norm(tf.gradients(self.LDA_HAN.loss_mse, tv_), self.clip)
        self.optimizer = optim.apply_gradients(tuple(zip(gradient_, tv_)), global_step=self.global_step)

    def train(self):
        loss_lst = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # sess.run([self.LDA_HAN.han_embed_init, self.LDA_HAN.lda_embed_init], feed_dict={self.LDA_HAN.han_embed_placeholder: self.embed_init_vec, self.LDA_HAN.lda_embed_placeholder: self.embed_init_vec})

            total_iters = 0
            for epoch in range(1, self.num_epochs):
                for iter_, (x, y, pivot_log, target_log, user_ids) in enumerate(self.train_loader.get_batch()):
                    total_iters += 1
                    feed_ = {self.LDA_HAN.x: x,
                             self.LDA_HAN.y: y,
                             self.LDA_HAN.pivot_idxs: pivot_log,
                             self.LDA_HAN.target_idxs: target_log,
                             self.LDA_HAN.doc_ids: user_ids}
                    _, step = sess.run([self.optimizer, self.global_step], feed_dict=feed_)
                    if total_iters % self.print_iters == 0:
                        ce_loss, lda_loss, mse_loss = sess.run([self.LDA_HAN.loss_ce, self.LDA_HAN.loss_lda, self.LDA_HAN.loss_mse], feed_dict=feed_)
                        loss_lst.append([ce_loss, lda_loss, mse_loss])
                        print("EPOCH [{}/{}], ITER [{}/{} ({})] \nLOSS:{:.4f}/{:.4f}/{:.4f}".format(epoch, self.num_epochs, iter_+1, (len(self.train_loader.log_seq) // self.batch_size), total_iters, ce_loss, lda_loss, mse_loss))

                    if total_iters % self.save_iters == 0:
                        self.saver.save(sess, os.path.join(self.save_path, 'LDA_HAN_{}iter.ckpt'.format(total_iters)), global_step=self.global_step)
                        np.save(os.path.join(self.save_path, 'loss_{}iter.ckpt'.format(total_iters)), np.array(loss_lst))
