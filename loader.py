import numpy as np
from operator import itemgetter


class log_dataloader(object):
    def __init__(self, log_seq, log_vocab, min_day, max_day, batch_size, target_hour):
        self.log_seq = log_seq
        self.log_vocab = log_vocab
        self.min_day = min_day
        self.max_day = max_day
        self.batch_size = batch_size
        self.target_len = target_hour * 6
        self.day_log = 24 * 6

    def log_shuffle(self, eval_=False):
        indices = np.arange(len(self.log_seq))
        np.random.shuffle(indices)
        log_seq = itemgetter(*indices)(self.log_seq)
        if eval_:
            log_seq = log_seq[:self.batch_size]
        return log_seq

    def random_log_sampling(self, user_log):
        day_length = np.random.randint(self.min_day, self.max_day+1)
        seq_size = day_length * self.day_log
        x_start = np.random.randint(0, len(user_log)-seq_size-self.target_len)
        x_end = x_start + seq_size

        subseq_x = user_log[x_start:x_end]
        subseq_x = subseq_x + [0] * (self.max_day * self.day_log - len(subseq_x))  # zero padding
        subseq_hx = [subseq_x[sub_i:(sub_i+self.day_log)] for sub_i in range(0, len(subseq_x), self.day_log)]  # to hierarchy

        y_end = x_end + self.target_len
        subseq_y = user_log[x_end:y_end]

        return subseq_hx, subseq_y

    def seq_to_ratio(self, y):
        count_ = np.unique(y, return_counts=True)
        ratio_ = [0] * len(self.log_vocab)
        for ch, cnt in zip(count_[0], count_[1]):
            ratio_[ch] = cnt/self.target_len
        return ratio_

    def get_batch(self, shuffle=True):
        if shuffle:
            log_seq = self.log_shuffle()
        else:
            log_seq = self.log_seq

        for user_i in range(0, len(log_seq)-self.batch_size, self.batch_size):
            batch_user = log_seq[user_i:(user_i+self.batch_size)]

            batch_hx, batch_y, user_ids = [], [], []
            for user_id, user_log in batch_user:
                subseq_hx, subseq_y = self.random_log_sampling(user_log)
                batch_hx.append(subseq_hx)
                batch_y.append(self.seq_to_ratio(subseq_y))
                user_ids.append(user_id)

            batch_hx = np.array(batch_hx)
            batch_y = np.array(batch_y)
            user_ids = np.array(user_ids)
            yield batch_hx, batch_y, user_ids

    def get_eval_batch(self):
        log_seq = self.log_shuffle(eval_=True)
        batch_hx, batch_y, user_ids = [], [], []
        for user_id, user_log in log_seq:
            subseq_hx, subseq_y = self.random_log_sampling(user_log)
            batch_hx.append(subseq_hx)
            batch_y.append(self.seq_to_ratio(subseq_y))
            user_ids.append(user_id)

        batch_hx = np.array(batch_hx)
        batch_y = np.array(batch_y)
        user_ids = np.array(user_ids)
        return batch_hx, batch_y, user_ids

    def test_loader(self):
        for user_id, user_log in self.log_seq:
            yield user_log, user_id
