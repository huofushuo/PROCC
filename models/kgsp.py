import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from sklearn.preprocessing import scale
from .common_kgsp import MLP
from .word_embedding import load_word_embeddings

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class KGSP(nn.Module):
    def __init__(self, dset, args):
        super(KGSP, self).__init__()
        self.obj_head = MLP(dset.feat_dim, 1024, 2, relu=True, dropout=True, norm=True, layers=[768, 1024])
        self.attr_head = MLP(dset.feat_dim, 1024, 2, relu=True, dropout=True, norm=True, layers=[768, 1024])
        self.attr_clf = MLP(1024, len(dset.attrs), 1, relu=False)
        self.obj_clf = MLP(1024, len(dset.objs), 1, relu=False)
        self.dset = dset
        self.args = args
        if dset.open_world:
            self.known_pairs = dset.train_pairs
            seen_pair_set = set(self.known_pairs)
            mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
            self.seen_mask = torch.BoolTensor(mask).cuda() * 1.

        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                          torch.arange(len(self.dset.objs)).long().to(device)
        self.feasibility_scores = self.feasibility_embeddings()

    def train_forward(self, x):
        img, attrs, objs, mask = x[0], x[1], x[2], x[4]

        attr_feats = self.attr_head(img)
        obj_feats = self.obj_head(img)

        attr_pred = self.attr_clf(attr_feats)
        obj_pred = self.obj_clf(obj_feats)

        if self.args.partial == True:
            attr_loss = F.cross_entropy(attr_pred[mask == 0, :], attrs[mask == 0])
            obj_loss = F.cross_entropy(obj_pred[mask == 1, :], objs[mask == 1])
        else:
            attr_loss = F.cross_entropy(attr_pred, attrs)
            obj_loss = F.cross_entropy(obj_pred, objs)

        if self.args.gumbel == True and self.args.partial == True:
            obj_inf = 1.0 * (torch.Tensor(self.feasibility_scores[:, attrs.cpu()]).cuda().permute(1, 0))
            obj_labels = F.gumbel_softmax(obj_inf * obj_pred, dim=-1, hard=True).argmax(-1).detach()
            obj_weak_loss = F.cross_entropy(obj_pred[mask == 0], obj_labels[mask == 0])

            att_inf = 1.0 * (torch.Tensor(self.feasibility_scores[objs.cpu(), :]).cuda())
            att_labels = F.gumbel_softmax(att_inf * attr_pred, dim=-1, hard=True).argmax(-1).detach()
            att_weak_loss = F.cross_entropy(attr_pred[mask == 1], att_labels[mask == 1])
            weak_loss = att_weak_loss + obj_weak_loss

        loss = attr_loss + obj_loss
        if self.args.gumbel == True and self.args.partial == True:
            loss = loss + weak_loss
        return loss, None

    def compute_feasibility(self):
        scores = np.load(self.args.kbfile, allow_pickle=True).item()
        feasibility_scores = [0 for i in range(len(self.dset.attrs) * len(self.dset.objs))]
        for a in self.dset.attrs:
            for o in self.dset.objs:
                score = scores[o][a]
                idx = self.dset.all_pair2idx[(a, o)]
                feasibility_scores[idx] = score

        self.feas_scores = feasibility_scores
        return feasibility_scores

    def feasibility_embeddings(self):

        scores = np.load(self.args.kbfile, allow_pickle=True).item()
        feasibility_scores = [[0 for i in range(len(self.dset.attrs))] for j in range(len(self.dset.objs))]
        for i in range(len(self.dset.objs)):
            for j in range(len(self.dset.attrs)):
                feasibility_scores[i][j] = max(scores[self.dset.objs[i]][self.dset.attrs[j]], 0)

        return np.array(feasibility_scores)

    def val_forward_with_threshold(self, x, th=0.):
        img = x[0]
        attr_pred = F.softmax(self.attr_clf(self.attr_head(img)), dim=1)
        obj_pred = F.softmax(self.obj_clf(self.obj_head(img)), dim=1)

        score = torch.bmm(attr_pred.unsqueeze(2), obj_pred.unsqueeze(1)).view(attr_pred.shape[0], -1)
        # Note: Pairs are already aligned here
        mask = torch.Tensor((np.array(self.feas_scores) >= th) * 1.0).cuda()
        score = score * mask + (1. - mask) * (-1.)
        scores = {}
        for itr, (attr, obj) in enumerate(self.dset.pairs):
            attr_id, obj_id = self.dset.attr2idx[attr], self.dset.obj2idx[obj]
            idx = obj_id + attr_id * len(self.dset.objs)
            scores[(attr, obj)] = score[:, idx]
        return score, scores

    def val_forward_revised(self, x):
        img = x[0]
        attr_pred = F.softmax(self.attr_clf(self.attr_head(img)), dim=1)
        obj_pred = F.softmax(self.obj_clf(self.obj_head(img)), dim=1)

        score = torch.bmm(attr_pred.unsqueeze(2), obj_pred.unsqueeze(1)).view(attr_pred.shape[0], -1)

        scores = {}
        for itr, (attr, obj) in enumerate(self.dset.pairs):
            attr_id, obj_id = self.dset.attr2idx[attr], self.dset.obj2idx[obj]
            idx = obj_id + attr_id * len(self.dset.objs)
            scores[(attr, obj)] = score[:, idx]
        return score, scores

    def forward(self, x, threshold=None):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                if threshold is None:
                    loss, pred = self.val_forward_revised(x)
                else:
                    loss, pred = self.val_forward_with_threshold(x, threshold)

        return loss, pred