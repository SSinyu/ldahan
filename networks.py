import numpy as np
import tensorflow as tf
from module import BiGRU, Attention_, fc_, dirichlet_likelihood


class HAN(object):
    def __init__(self, vocab_size, embed_size, pre_word=True, word_tuning=True, word_init_vec=None):
        assert embed_size % 2 == 0
        tf.reset_default_graph()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = embed_size // 2
        self.max_sent_length = 144
        self.max_doc_length = 10
        
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.x = tf.placeholder(tf.int32, shape=[None, None, None], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, vocab_size], name='y')

        if pre_word:
            self.word_init = tf.get_variable(shape=[self.vocab_size, self.embed_size], initializer=tf.constant_initializer(word_init_vec), trainable=word_tuning, name='word_embed')
        else:
            self.word_init = tf.Variable(tf.truncated_normal((self.vocab_size, self.embed_size)), trainable=word_tuning, name='word_embed')

        with tf.variable_scope('HAN'):
            self.doc_vec = self.han(self.word_init)

        with tf.variable_scope('mse_loss'):
            out = fc_(self.doc_vec, self.vocab_size)
            out = tf.nn.softmax(out)
            self.loss_mse = tf.reduce_mean(tf.square(self.y-out))

    def han(self, han_embed):
        with tf.variable_scope('word2vec'):
            word_embed = tf.nn.embedding_lookup(han_embed, self.x)
        with tf.variable_scope('sent2vec'):
            word_vector = tf.reshape(word_embed, shape=[-1, self.max_sent_length, self.embed_size])
            sent_encode = BiGRU(word_vector, self.hidden_size)
            sent_attn = Attention_(sent_encode, self.hidden_size * 2)
        with tf.variable_scope('doc2vec'):
            sent_vector = tf.reshape(sent_attn, shape=[-1, self.max_doc_length, self.hidden_size*2])
            doc_encode = BiGRU(sent_vector, self.hidden_size)
            doc_attn = Attention_(doc_encode, self.hidden_size*2)
        return doc_attn


class LDAHAN(object):
    def __init__(self, n_documents, vocab_size, embed_size, n_topics, pre_embed=True, embed_tuning=True, embed_init_vec=None, pre_doc=True, doc_tuning=True, doc_init_vec=None, lda_dropout=0.3, lambda_=200, temperature=10.0, c='add'):
        assert embed_size % 2 == 0
        tf.reset_default_graph()

        self.n_documents = n_documents
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = embed_size // 2
        self.n_topics = n_topics
        self.pre_embed = pre_embed
        self.embed_tuning = embed_tuning
        self.pre_doc = pre_doc
        self.doc_tuning = doc_tuning
        self.doc_init_vec = doc_init_vec
        self.lda_dropout = lda_dropout
        self.lambda_ = lambda_
        self.temperature = temperature
        self.max_sent_length = 144
        self.max_doc_length = 10

        # HAN placeholder
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.x = tf.placeholder(tf.int32, shape=[None, None, None], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, vocab_size], name='y')
        # LDA placeholder
        self.doc_ids = tf.placeholder(tf.int32, shape=[None,], name='doc_ids')

        # word vector
        if pre_embed:
            self.word_init = tf.get_variable(shape=[self.vocab_size, self.embed_size], initializer=tf.constant_initializer(embed_init_vec), name='word_embed', trainable=self.embed_tuning)
        else:
            self.word_init = tf.Variable(tf.truncated_normal((self.vocab_size, self.embed_size)), trainable=self.embed_tuning, name='word_embed')

        # HAN
        with tf.variable_scope('HAN'):
            self.doc_vec_HAN = self.han(self.word_init)

        # LDA2vec
        with tf.variable_scope('LDA2vec'):
            self.doc_weight, self.doc_vec_LDA, self.topic_matrix = self.lda2vec()
            self.doc_vec_LDA = tf.nn.dropout(self.doc_vec_LDA, self.lda_dropout)

        # dirichlet loss
        with tf.variable_scope('lda_loss'):
            alpha = 1/self.n_topics
            self.loss_lda = dirichlet_likelihood(self.doc_weight, self.n_topics, alpha) * self.lambda_

        # mse loss
        with tf.variable_scope('mse_loss'):
            if c == 'add':
                merged_doc_vector = tf.add(self.doc_vec_LDA, self.doc_vec_HAN)
            elif c == 'concat':
                merged_doc_vector = tf.concat([self.doc_vec_LDA, self.doc_vec_HAN], axis=1)
            elif c == 'wsum':
                merged_doc_vector = tf.concat([self.doc_vec_LDA, self.doc_vec_HAN], axis=1)
                merged_doc_vector = fc_(merged_doc_vector, self.embed_size)
            else:
                raise ValueError

            out = fc_(merged_doc_vector, self.vocab_size)
            out = tf.nn.softmax(out)
            self.loss_mse = tf.reduce_mean(tf.square(self.y-out))

    def han(self, han_embed):
        with tf.variable_scope('word2vec'):
            word_embed = tf.nn.embedding_lookup(han_embed, self.x)
        # sentence encoding
        with tf.variable_scope('sent2vec'):
            word_vector = tf.reshape(word_embed, shape=[-1, self.max_sent_length, self.embed_size])
            sent_encode = BiGRU(word_vector, self.hidden_size)
            sent_attn = Attention_(sent_encode, self.hidden_size * 2)
        # document encoding
        with tf.variable_scope('doc2vec'):
            sent_vector = tf.reshape(sent_attn, shape=[-1, self.max_doc_length, self.hidden_size*2])
            doc_encode = BiGRU(sent_vector, self.hidden_size)
            doc_attn = Attention_(doc_encode, self.hidden_size * 2)
        return doc_attn

    def lda2vec(self, init_=True):
        scalar = 1 / np.sqrt(self.n_documents + self.n_topics)
        if init_:
            doc_weight = tf.get_variable(shape=[self.n_documents, self.n_topics], initializer=tf.constant_initializer(self.doc_init_vec), name='doc_init', trainable=True)
        else:
            doc_weight = tf.Variable(tf.random_normal([self.n_documents, self.n_topics], mean=0, stddev=50*scalar), trainable=True)
        doc_weight_embed = tf.nn.embedding_lookup(doc_weight, self.doc_ids)

        doc_proportions = tf.nn.softmax(doc_weight_embed/self.temperature)
        topic_matrix = tf.get_variable('topics', shape=(self.n_topics, self.embed_size), dtype=tf.float32, initializer=tf.orthogonal_initializer(gain=scalar), trainable=True)
        topic_matrix = tf.nn.dropout(topic_matrix, self.lda_dropout)
        doc_vector = tf.matmul(doc_proportions, topic_matrix)

        return doc_weight_embed, doc_vector, topic_matrix

