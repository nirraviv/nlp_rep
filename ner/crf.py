from torch import nn
import torch


def log_sum_exp_batch(vec, axis=-1):
    # vec shape (batch_size, n, m)
    max_score = torch.max(vec, axis)[0]
    max_score_broadcast = max_score.view(vec.shape[0], -1, 1)
    return max_score + torch.logsumexp(vec - max_score_broadcast, axis)


class CRF(nn.Module):
    def __init__(self, start_label_id, stop_label_id, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.num_labels, self.num_labels))
        # These two statements enforce the constraint that we never transfer *to* the start tag(or label),
        # and we never transfer *from* the stop label (the model would probably learn this anyway,
        # so this enforcement is likely unimportant)
        self.transitions.data[start_label_id, :] = -10000.
        self.transitions.data[:, stop_label_id] = -10000.

    def _viterbi_decode(self, feats):
        """ Max-product algorithm or viterbi"""
        batch_size, seq_length = feats.shape[:2]

        log_delta = feats.new_full((batch_size, 1, self.num_labels), -10000.)
        log_delta[:, 0, self.start_label_id] = 0

        # psi = max P(this_latent)
        psi = feats.new_zeros((batch_size, seq_length, self.num_labels)).long()
        for t in range(1, seq_length):
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = feats.new_zeros((batch_size, seq_length)).long()

        # max p(z1:t, all_x | theta)
        max_log_delta, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(seq_length-2, -1, -1):
            # choose the state of z_t according to the state of z_(t+1)
            path[:, t] = psi[:, t+1].gather(-1, path[:, t+1].view(-1, 1)).squeeze()

        return max_log_delta, path

    def _forward_alg(self, feats):
        """ alpha-recursion to calculate log_prob of all barX"""
        batch_size, seq_length = feats.shape[:2]

        # alpha recursion alpha(zt) = p(zt, bar_x_1:t)
        log_alpha = feats.new_full((batch_size, 1, self.num_labels), -10000.)
        log_alpha[:, 0, self.start_label_id] = 0

        for t in range(1, seq_length):
            log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)

        log_prob_all_bar_x = log_sum_exp_batch(log_alpha)
        return log_prob_all_bar_x

    def _score_sentence(self, feats, label_ids):
        """ score of provided label sequence """
        batch_size, seq_length = feats.shape[:2]

        batch_transitions = self.transitions.expand(batch_size, self.num_labels, self.num_labels)
        batch_transitions = batch_transitions.flatten(1)

        score = feats.new_zeros((batch_size, 1))
        for t in range(1, seq_length):
            score += batch_transitions.gather(-1, (label_ids[:, t]*self.num_labels+label_ids[:, t-1]).view(-1, 1)) + \
                feats[:, t].gather(-1, label_ids[:, t].view(-1, 1)).view(-1, 1)
        return score

    def _neg_log_likelihood(self, feats, label_ids):
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, label_ids)
        return torch.mean(forward_score - gold_score)

    def forward(self, features, label_ids=None):
        if self.training:
            return self._neg_log_likelihood(features, label_ids)
        else:
            return self._viterbi_decode(features)
