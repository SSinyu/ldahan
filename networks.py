import numpy as np
import tensorflow as tf
from module import BiGRU, Attention_, fc_, dirichlet_likelihood, cross_entropy


class LDA2vec_HAN(object):
    def __init__(self, n_documents, vocab_size, embed_size, n_topics, pre_embed=True, embed_tuning=True, lda_dropout=0.3, lambda_=200, temperature=1.0):
        assert embed_size % 2 == 0
        tf.reset_default_graph()

        self.n_documents = n_documents
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = embed_size // 2
        self.n_topics = n_topics
        self.pre_embed = pre_embed
        self.embed_tuning = embed_tuning
        self.lda_dropout = lda_dropout
        self.lambda_ = lambda_
        self.temperature = temperature
        self.max_sent_length = 144
        self.max_doc_length = 10

        # HAN placeholder
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.x = tf.placeholder(tf.int32, shape=[None, None, None], name='x')
        self.y = tf.placeholder(tf.int32, shape=[None, vocab_size], name='y')
        # LDA placeholder
        self.pivot_idxs = tf.placeholder(tf.int32, shape=[None,], name='pivot_idxs')
        self.target_idxs = tf.placeholder(tf.int32, shape=[None,], name='taregt_idxs')
        self.doc_ids = tf.placeholder(tf.int32, shape=[None,], name='doc_ids')

        # word vector
        if pre_embed:
            han_embed = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.embed_size]), trainable=self.embed_tuning, name='word_embed_HAN')
            lda_embed = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.embed_size]), trainable=self.embed_tuning, name='word_embed_LDA')
            # self.han_embed_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embed_size])
            # self.lda_embed_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embed_size])
            # self.han_embed_init = han_embed.assign(self.han_embed_placeholder)
            # self.lda_embed_init = lda_embed.assign(self.lda_embed_placeholder)
            embed_init_vec = np.load('/home/datamininglab/Downloads/sinyu/embed.npy')
            self.han_embed_init = tf.get_variable(name='word_embed_HAN', shape=[self.vocab_size, self.embed_size], initializer=tf.constant_initializer(embed_init_vec))
            self.lda_embed_init = tf.get_variable(name='word_embed_LDA', shape=[self.vocab_size, self.embed_size], initializer=tf.constant_initializer(embed_init_vec))
        else:
            self.han_embed_init = tf.Variable(tf.truncated_normal((self.vocab_size, self.embed_size)), trainable=self.embed_tuning, name='word_embed_HAN')
            init_width = 0.5 / self.embed_size
            self.lda_embed_init = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -init_width, init_width), name='word_embed_LDA')

        # HAN
        with tf.variable_scope('HAN'):
            doc_vec_HAN = self.HAN(self.han_embed_init)

        # LDA2vec
        with tf.variable_scope('LDA2vec'):
            doc_weight, word_vec_LDA, doc_vec_LDA, topic_matrix = self.LDA2vec(self.lda_embed_init)
            context = (tf.nn.dropout(word_vec_LDA, self.lda_dropout),
                       tf.nn.dropout(doc_vec_LDA, self.lda_dropout))
            context_vector = tf.add(*context)

        # nce loss
        with tf.variable_scope('cross_entropy_loss'):
            self.loss_ce = cross_entropy(word_vec_LDA, self.target_idxs, self.vocab_size)

        # dirichlet loss
        with tf.variable_scope('lda_loss'):
            alpha = 1/self.n_topics
            self.loss_lda = dirichlet_likelihood(doc_weight, self.n_topics, alpha) * self.lambda_

        # mse loss
        with tf.variable_scope('mse_loss'):
            merged_doc_vector = tf.add(doc_vec_LDA, doc_vec_HAN)
            out = fc_(merged_doc_vector, self.vocab_size)
            self.loss_mse = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=out))


    def HAN(self, han_embed):
        with tf.variable_scope('word2vec'):
            word_embed = tf.nn.embedding_lookup(han_embed, self.x)
            word_embed = tf.reshape(word_embed, shape=[-1, self.max_sent_length, self.embed_size])
        # sentence encoding
        with tf.variable_scope('sent2vec'):
            sent_encode = BiGRU(word_embed, self.hidden_size)
            sent_attn = Attention_(sent_encode, self.hidden_size * 2)
        # document encoding
        with tf.variable_scope('doc2vec'):
            sent_vector = tf.reshape(sent_attn, shape=[-1, self.max_doc_length, self.hidden_size*2])
            doc_encode = BiGRU(sent_vector, self.hidden_size)
            doc_attn = Attention_(doc_encode, self.hidden_size * 2)
        return doc_attn


    def LDA2vec(self, lda_embed):
        word_embed = tf.nn.embedding_lookup(lda_embed, self.pivot_idxs)

        # document
        scalar = 1 / np.sqrt(self.n_documents + self.n_topics)
        doc_weight = tf.Variable(tf.random_normal([self.n_documents, self.n_topics], mean=0, stddev=50*scalar))
        #unique_doc_ids, index_doc_ids = tf.unique(self.doc_ids)
        #index_doc_ids = tf.unique_with_counts(index_doc_ids).count
        #doc_weight_embed = tf.nn.embedding_lookup(doc_weight, unique_doc_ids)
        doc_weight_embed = tf.nn.embedding_lookup(doc_weight, self.doc_ids)

        doc_proportions = tf.nn.softmax(doc_weight_embed/self.temperature)
        topic_matrix = tf.get_variable('topics', shape=(self.n_topics, self.embed_size), dtype=tf.float32, initializer=tf.orthogonal_initializer(gain=scalar))
        topic_matrix = tf.nn.dropout(topic_matrix, self.lda_dropout)
        doc_vector = tf.matmul(doc_proportions, topic_matrix)

        return doc_weight_embed, word_embed, doc_vector, topic_matrix

