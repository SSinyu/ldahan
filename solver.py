import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from networks import LDAHAN, HAN


class Solver(object):
    def __init__(self, args, data_loader=None):
        self.data_loader = data_loader

        self.save_path = args.save_path
        self.n_user = len(data_loader.log_seq)

        self.vocab_size = len(data_loader.log_vocab)
        self.n_topics = args.n_topics
        self.batch_size = args.batch_size

        if args.pre_embed:
            self.embed_init_vec = np.load(os.path.join(args.data_path, 'word_init.npy'))
            self.embed_size = self.embed_init_vec.shape[1]
        else:
            self.embed_size = args.embed_size
        self.embed_tuning = args.embed_tuning
        # hidden_size = embed_size/2

        if args.pre_doc:
            self.doc_init_vec = np.load(os.path.join(args.data_path, 'doc_init.npy'))
        self.doc_tuning = args.doc_tuning

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.eval_iters = args.eval_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters

        self.clip = args.clip
        self.lambda_ = args.lambda_

        self.m = args.m

        if self.m == 'ldahan':
            self.model = LDAHAN(self.n_user, self.vocab_size, self.embed_size, self.n_topics, pre_embed=args.pre_embed, embed_tuning=self.embed_tuning, embed_init_vec=self.embed_init_vec, pre_doc=args.pre_doc, doc_tuning=self.doc_tuning, doc_init_vec=self.doc_init_vec, lambda_=self.lambda_)
            self.saver = tf.train.Saver(tf.global_variables())
            self.saver.export_meta_graph(os.path.join(self.save_path, 'LDA_HAN_graph.meta'))
        elif self.m == 'han':
            self.model = HAN(self.vocab_size, self.embed_size, args.pre_embed, self.embed_tuning, self.embed_init_vec)
            self.saver = tf.train.Saver(tf.global_variables())
            self.saver.export_meta_graph(os.path.join(self.save_path, 'HAN_graph.meta'))
        else:
            raise ValueError

        self.global_step = tf.Variable(0, trainable=False)

        # learning rate decay
        lr_decay = tf.train.exponential_decay(args.lr, self.global_step, self.decay_iters, 0.95)

        # gradient clipping
        optim = tf.train.AdamOptimizer(lr_decay)
        tv_ = tf.trainable_variables()
        gradient_, _ = tf.clip_by_global_norm(tf.gradients(self.model.loss_mse, tv_), self.clip)
        self.optimizer = optim.apply_gradients(tuple(zip(gradient_, tv_)), global_step=self.global_step)

    def make_feed(self, x, y, user_ids):
        if self.m == 'ldahan':
            feed_ = {self.model.x: x,
                     self.model.y: y,
                     self.model.doc_ids: user_ids}
        else:
            feed_ = {self.model.x: x,
                     self.model.y: y}
        return feed_

    def save_array(self, f_name, f):
        np.save(os.path.join(self.save_path, f_name), f)

    def train(self):
        train_loss_lst, eval_loss_lst = [], []
        start_time = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            total_iters = 0
            for epoch in range(1, self.num_epochs):
                for iter_, (x, y, user_ids) in enumerate(self.data_loader.get_batch()):
                    total_iters += 1
                    feed_ = self.make_feed(x, y, user_ids)
                    if self.m == 'ldahan':
                        _, step, lda_loss, mse_loss = sess.run([self.optimizer, self.global_step, self.model.loss_lda, self.model.loss_mse], feed_dict=feed_)
                    else:
                        _, step, mse_loss = sess.run([self.optimizer, self.global_step, self.model.loss_mse], feed_dict=feed_)
                    train_loss_lst.append([(lda_loss if self.m == 'ldahan' else 0), mse_loss])

                    # print out
                    if total_iters % self.print_iters == 0:
                        print("EPOCH [{}/{}], ITER [{}/{} ({})], TIME :{:.1f}\nLOSS:{:.4f}/{:.4f}".format(epoch, self.num_epochs, iter_+1, (len(self.data_loader.log_seq) // self.batch_size), total_iters, time.time() - start_time, (lda_loss if self.m == 'ldahan' else 0), mse_loss))

                    # evaluation
                    if total_iters % self.eval_iters == 0:
                        e_x, e_y, e_ids = self.data_loader.get_eval_batch()
                        feed_ = self.make_feed(e_x, e_y, e_ids)
                        if self.m == 'ldahan':
                            e_lda_loss, e_mse_loss = sess.run([self.model.loss_lda, self.model.loss_mse], feed_dict=feed_)
                        else:
                            e_mse_loss = sess.run(self.model.loss_mse, feed_dict=feed_)
                        eval_loss_lst.append([(e_lda_loss if self.m == 'ldahan' else 0), e_mse_loss])
                        print('==EVALUATION LOSS:{:.4f}/{:.4f}'.format((e_lda_loss if self.m == 'ldahan' else 0), e_mse_loss))

                    # save
                    if total_iters % self.save_iters == 0:
                        if self.m == 'ldahan':
                            self.saver.save(sess, os.path.join(self.save_path, 'ldahan.ckpt'), global_step=self.global_step)
                            topic_vec = sess.run(self.model.topic_matrix)
                            self.save_array('topic_vec_{}iter'.format(total_iters), topic_vec)


                        else:
                            self.saver.save(sess, os.path.join(self.save_path, 'han.ckpt'), global_step=self.global_step)

                        self.save_array('train_loss_{}iter'.format(total_iters), np.array(train_loss_lst))
                        self.save_array('eval_loss_{}iter'.format(total_iters), np.array(eval_loss_lst))
                        self.save_fig(np.array(train_loss_lst), np.array(eval_loss_lst))

    def test(self):
        del self.model
        del self.saver
        tf.reset_default_graph()

        if self.m == 'ldahan':
            self.model = LDAHAN(self.n_user, self.vocab_size, self.embed_size, self.n_topics)
            f = os.path.join(self.save_path, 'ldahan.ckpt-{}'.format(self.test_iters))
        elif self.m == 'han':
            self.model = HAN(self.vocab_size, self.embed_size,)
            f = os.path.join(self.save_path, 'han.ckpt-{}'.format(self.test_iters))
        else:
            raise ValueError

        self.saver = tf.train.Saver()

        with tf.Session() as sess:
            self.saver.restore(sess, f)

            doc_v = np.array([])
            for i, (user_log, user_id) in enumerate(self.data_loader.test_loader):
                if self.m == 'ldahan':
                    feed_ = {self.model.x:user_log,
                             self.model.doc_ids:user_id}
                    doc_ = sess.run(self.model.doc_vec_HAN, feed_dict=feed_)
                else:
                    feed_ = {self.model.x:user_log}
                    doc_ = sess.run(self.model.doc_vec, feed_dict=feed_)

                doc_v = np.concatenate((doc_v, doc_))



    def save_fig(self, train_loss, eval_loss):
        if self.m == 'ldahan':
            fig, ax = plt.subplots(2, 1, figsize=(20, 16))
            sns.lineplot(range(len(train_loss)), train_loss[:, 0], ax=ax[0])
            sns.lineplot(range(len(eval_loss)), eval_loss[:, 0], ax=ax[0].twiny(), color='purple')
            ax[0].set_title('Dirichlet loss', fontsize=20)
            sns.lineplot(range(len(train_loss)), train_loss[:, 1], ax=ax[1])
            sns.lineplot(range(len(eval_loss)), eval_loss[:, 1], ax=ax[1].twiny(), color='purple')
            ax[1].set_title('MSE loss', fontsize=20)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(20, 8))
            sns.lineplot(range(len(train_loss)), train_loss[:, 1], ax=ax)
            sns.lineplot(range(len(eval_loss)), eval_loss[:, 1], ax=ax.twiny(), color='purple')
            ax.set_title('MSE_loss', fontsize=20)

        plt.savefig(os.path.join(self.save_path, 'loss_fig.png'))


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill=' '):
    # by https://gist.github.com/snakers4/91fa21b9dda9d055a02ecd23f24fbc3d
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '=' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()
