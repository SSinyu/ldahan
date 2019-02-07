import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from networks import lda2vec_HAN_ver1


class Solver(object):
    def __init__(self, config, train_loader=None, test_loader=None):
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_path = config.input_path
        self.save_path = config.save_path

        self.vocab_size = 24
        self.n_topics = config.n_topics
        self.per_user = config.per_user
        self.n_user = config.n_user
        self.min_day_length = config.min_day_length
        self.max_day_length = config.max_day_length

        self.num_epochs = config.num_epochs
        self.decay_iters = config.decay_iters
        self.print_iters = config.print_iters
        self.save_iters = config.save_iters
        self.test_iters = config.test_iters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.lambda_ = config.lambda_
        self.alpha_w = config.alpha_w
        self.alpha_ = 1/self.n_topics * config.alpha_w

        self.hidden_size = config.hidden_size
        self.lda_han = lda2vec_HAN_ver1(n_topics=self.n_topics,vocab_size=self.vocab_size, embed_size=self.hidden_size, hidden_size=self.hidden_size, embed_fine_tune=True)
        self.lda_han.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.lda_han.parameters(), config.lr, [self.beta1, self.beta2])

    def save_model(self, iter_):
        f = os.path.join(self.save_path, "ldahan_{}iter.ckpt".format(iter_))
        torch.save(self.lda_han.state_dict(), f)

    def load_model(self, iter_, multigpu_=None):
        f = os.path.join(self.save_path, "ldahan_{}iter.ckpt".format(iter_))
        if multigpu_:
            state_d = OrderedDict()
            for k,v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.lda_han.load_state_dict(state_d)
        else:
            self.lda_han.load_state_dict(torch.load(f))

    def prep_input(self, x_, y_, doc_len_):
        # reshape
        doc_len_ = doc_len_.view(-1).to(self.device)
        x_ = x_.view(-1, max(doc_len_).item(), 144).to(self.device)
        y_ = y_.view(-1, self.vocab_size).float().to(self.device)
        # sort lengths
        doc_len_, perm_idx = doc_len_.sort(0, descending=True)
        x_, y_ = x_[perm_idx], y_[perm_idx]
        # sent lengths is fixed
        sent_len_ = (torch.ones(x_.shape[0] * x_.shape[1]) * 144).long()
        return x_, y_, sent_len_, doc_len_

    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        dirichlet_, mse_ = [], []
        total_iters = 0
        for epoch in range(1, self.num_epochs+1):
            self.lda_han.train(True)

            for iter_, (log_x, log_y, doc_lengths) in enumerate(self.train_loader):
                total_iters += 1
                log_x, log_y, sent_lengths, doc_lengths = self.prep_input(log_x, log_y, doc_lengths)
                doc_prop, pred_prop, _ = self.lda_han(log_x, sent_lengths, doc_lengths)

                dirichlet_loss = doc_prop.sum(1).mean()
                #dirichlet_loss = ((self.alpha_ -1) * doc_prop).sum(1).mean()
                dirichlet_loss *= self.lambda_

                mse_loss = self.criterion(pred_prop, log_y)
                loss = dirichlet_loss + mse_loss

                self.lda_han.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                dirichlet_.append(dirichlet_loss.item())
                mse_.append(mse_loss.item())

                if total_iters % self.print_iters == 0:
                    print("EPOCH [{}/{}], ITER [{}/{} ({})] \nLOSS: {:.4f} \nDirichlet: {:.4f}, MSE: {:.4f}".format(epoch, self.num_epochs, iter_+1, len(self.train_loader), total_iters, loss.item(), dirichlet_loss.item(), mse_loss.item()))

                ## learning rate decay
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()

                ## save
                if total_iters % self.save_iters == 0:
                    self.save_model(total_iters)
                    np.save(os.path.join(self.save_path, 'loss{}_iter.npy'.format(total_iters)), np.column_stack((dirichlet_, mse_)))

            ## save
            torch.save(self.lda_han.state_dict(), os.path.join(self.save_path, "ldahan_{}ep.ckpt".format(epoch)))
            np.save(os.path.join(self.save_path, 'loss{}ep.npy'.format(epoch)), np.column_stack((dirichlet_, mse_)))

    def test(self):
        self.load_model(self.test_iters)

        # topic vector
        state = torch.load(os.path.join(self.save_path, "ldahan_{}iter.ckpt".format(self.test_iters)))
        topic_vector = state['topic_matrix']
        np.save(os.path.join(self.save_path, 'result_topic_vector_{}.npy'.format(self.test_iters)), topic_vector.cpu().numpy())

        # doc vector
        doc_vector = np.array([]).reshape(0,400)
        for i, (test_x, test_doc_len) in enumerate(self.test_loader):
            test_x = test_x.to(self.device)
            test_doc_len = test_doc_len.view(-1).to(self.device)
            test_sent_len = (torch.ones(test_x.shape[0] * test_x.shape[1]) * 144).long().to(self.device)

            _, _, doc_v = self.lda_han(test_x, test_sent_len, test_doc_len)
            doc_v = doc_v.cpu().detach().numpy()
            doc_vector = np.concatenate((doc_vector, doc_v))

            printProgressBar(i, len(self.test_loader), prefix='calculate doc vector..', suffix='Complete', length=25)
        np.save(os.path.join(self.save_path, 'result_doc_vector_{}.npy'.format(self.test_iters)), doc_vector)


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill=' '):
    # by https://gist.github.com/snakers4/91fa21b9dda9d055a02ecd23f24fbc3d
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '=' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()

