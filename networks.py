import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy.stats import ortho_group


class lda2vec_HAN_ver1(nn.Module):
    def __init__(self, n_topics, vocab_size, embed_size, hidden_size, pre_embed=None, embed_fine_tune=True):
        super(lda2vec_HAN_ver1, self).__init__()
        self.HAN_hidden_size = hidden_size
        self.LDA_hidden_size = hidden_size*2 # bidirectional encoding in HAN
        self.HAN = HierarchicalAttentionNet(vocab_size, embed_size, self.HAN_hidden_size, pre_embed, embed_fine_tune)

        # dirichlet loss
        assert n_topics < hidden_size
        self.n_topics = n_topics
        #topic_matrix = ortho_group.rvs(self.LDA_hidden_size)[0:self.n_topics]
        #self.topic_matrix = nn.Parameter(torch.FloatTensor(topic_matrix), requires_grad=True) # (hidden_size*2, n_topics)

        self.topic_matrix = nn.Parameter(torch.empty(self.n_topics, self.LDA_hidden_size), requires_grad=True)
        nn.init.orthogonal_(self.topic_matrix)

        # mse loss
        self.linear_ = nn.Linear(self.HAN_hidden_size*2, vocab_size)

    def forward(self, x, sent_len, doc_len):
        doc_vector = self.HAN(x, sent_len, doc_len) # (batch, hidden_size*2)

        # dirichlet loss
        doc_weight = torch.mm(doc_vector, torch.t(self.topic_matrix)) # (batch, n_topics)
        doc_proportion = F.log_softmax(doc_weight, dim=1)

        # mse loss
        pred_proportion = F.softmax(self.linear_(doc_vector), dim=1) # (batch, vocab_size)

        return doc_proportion, pred_proportion, doc_vector


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size=200):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size * 2
        self.linear_ = nn.Linear(self.hidden_size, self.hidden_size)
        self.tanh_ = nn.Tanh()
        self.softmax_ = nn.Softmax(dim=1)

    def forward(self, x):
        u_context = torch.nn.Parameter(torch.FloatTensor(self.hidden_size).normal_(0, 0.01)).cuda()
        h = self.tanh_(self.linear_(x)).cuda()
        alpha = self.softmax_(torch.mul(h, u_context).sum(dim=2, keepdim=True))  # (x_dim0, x_dim1, 1)
        attention_output = torch.mul(x, alpha).sum(dim=1)  # (x_dim0, x_dim2)
        return attention_output, alpha


class HierarchicalAttentionNet(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, pre_embed=None, embed_finetune=True):
        super(HierarchicalAttentionNet, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        # word level
        self.embed = nn.Embedding(vocab_size, embed_size)
        if pre_embed:
            self.weight = nn.Parameter(pre_embed, requires_grad=(True if embed_finetune else False))
        else:
            init.normal_(self.embed.weight, std=0.01)
            self.embed.weight.requires_grad = True
        self.word_rnn = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self.word_att = AttentionLayer(hidden_size)
        # sent level
        self.sent_rnn = nn.GRU(hidden_size*2, hidden_size, bidirectional=True)
        self.sent_att = AttentionLayer(hidden_size)

    def forward(self, x, sent_lengths, doc_lengths):
        # x shape -> (batch, max_doc, max_sent)
        # 'sent_lengths', 'doc_lengths' must sorted in decreasing order

        assert x.shape[2] == max(sent_lengths)
        assert x.shape[1] == max(doc_lengths)
        max_sent_len = x.shape[2]
        max_doc_len = x.shape[1]

        # word embedding
        word_embed = self.embed(x) # (batch, max_doc, max_sent, embed_size)

        # word encoding
        word_embed = word_embed.view(-1, max_sent_len, self.embed_size)  # (-1(batch), max_sent_len, embed_size)
        word_packed_input = pack_padded_sequence(word_embed, sent_lengths.cpu().numpy(), batch_first=True)
        word_packed_output, _ = self.word_rnn(word_packed_input)
        word_encode, _ = pad_packed_sequence(word_packed_output, batch_first=True) # (batch, max_sent_len, hidden_size*2)

        # word attention
        sent_vector, sent_alpha = self.word_att(word_encode) # (batch, hidden_size*2)

        # sent encoding
        sent_vector = sent_vector.view(-1, max_doc_len, self.hidden_size*2) # (batch, max_doc_len, hidden_size*2)
        sent_packed_input = pack_padded_sequence(sent_vector, doc_lengths.cpu().numpy(), batch_first=True)
        sent_packed_output, _ = self.sent_rnn(sent_packed_input)
        sent_encode, _ = pad_packed_sequence(sent_packed_output, batch_first=True) # (batch, max_doc_len, hidden_size*2)

        # sent attention
        doc_vector, doc_alpha = self.sent_att(sent_encode) # (batch, hidden_size*2)

        return doc_vector
