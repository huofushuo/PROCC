import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from sklearn.preprocessing import scale
from .common import MLP

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class PROCC(nn.Module):
    def __init__(self, dset, args):
        super(PROCC, self).__init__()

        self.obj_head1 = MLP(dset.feat_dim_h, 768, 1, relu=True, dropout=True, norm=True)
        self.obj_head2 = MLP(768, 512, 1, relu=True, dropout=True, norm=True)
        self.attr_head1 = MLP(dset.feat_dim_l, 768, 1, relu=True, dropout=True, norm=True)
        self.attr_head2 = MLP(768, 512, 1, relu=True, dropout=True, norm=True)
        self.obj_clf = MLP(512, len(dset.objs), 1, relu=False)
        self.attr_clf = MLP(512, len(dset.attrs), 1, relu=False)
        self.cross_attention_logit1 = cross_attention(77)  # obj to attr (attr, obj)
        self.cross_attention1 = cross_attention(51)       #obj to attr (attr, obj)
        self.cross_attention_logit2 = cross_attention(77)   # attr to obj (obj, attr)
        self.cross_attention2 = cross_attention(51)         #attr to obj (obj, attr)

        self.dset = dset
        self.args = args
        if dset.open_world:
            self.known_pairs = dset.train_pairs
            seen_pair_set = set(self.known_pairs)
            mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
            self.seen_mask = torch.BoolTensor(mask).cuda() * 1.

        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                          torch.arange(len(self.dset.objs)).long().to(device)


    def train_forward(self, x, step):
        img_attr, img_obj, attrs, objs, mask = x[0], x[1], x[2], x[3], x[5]

        if step == 'obj':
            freeze(self.attr_head1)
            freeze(self.attr_head2)
            freeze(self.attr_clf)
            active(self.obj_head1)
            active(self.obj_head2)
            active(self.obj_clf)

            obj_feats1 = self.obj_head1(img_obj)
            obj_feats2 = self.obj_head2(obj_feats1)
            obj_pred = self.obj_clf(obj_feats2)

            if self.args.partial == True:
                obj_loss = F.cross_entropy(obj_pred[mask == 1, :], objs[mask == 1])
            else:
                obj_loss = F.cross_entropy(obj_pred, objs)

            loss = obj_loss

        elif step == 'attr':
            freeze(self.obj_head1)
            freeze(self.obj_head2)
            freeze(self.obj_clf)
            active(self.attr_head1)
            active(self.attr_head2)
            active(self.attr_clf)
            active(self.cross_attention1)
            active(self.cross_attention_logit1)
            ############################################################################
            obj_feats1 = self.obj_head1(img_obj)
            attr_feats1 = self.attr_head1(img_attr)
            obj_feats2 = self.obj_head2(obj_feats1)
            attr_feats2 = self.attr_head2(self.cross_attention_logit1(attr_feats1, obj_feats1))
            attr_pred = self.attr_clf(self.cross_attention1(attr_feats2, obj_feats2))

            if self.args.partial == True:
                attr_loss = F.cross_entropy(attr_pred[mask == 0, :], attrs[mask == 0])
            else:
                attr_loss = F.cross_entropy(attr_pred, attrs)

            loss = attr_loss

        elif step == 'normal':

            active(self.obj_head1)
            active(self.obj_head2)
            active(self.obj_clf)
            active(self.attr_head1)
            active(self.attr_head2)
            active(self.attr_clf)
            active(self.cross_attention1)
            active(self.cross_attention2)
            active(self.cross_attention_logit1)
            active(self.cross_attention_logit2)
            ############################################################################
            attr_feats1 = self.attr_head1(img_attr)
            obj_feats1 = self.obj_head1(img_obj)

            attr_feats2 = self.attr_head2(self.cross_attention_logit1(attr_feats1, obj_feats1))
            obj_feats2 = self.obj_head2(self.cross_attention_logit2(obj_feats1, attr_feats1))

            attr_pred = self.attr_clf(self.cross_attention1(attr_feats2, obj_feats2))
            obj_pred = self.obj_clf(self.cross_attention2(obj_feats2, attr_feats2))

            if self.args.partial == True:
                attr_loss = F.cross_entropy(attr_pred[mask == 0, :], attrs[mask == 0])
                obj_loss = F.cross_entropy(obj_pred[mask == 1, :], objs[mask == 1])
            else:
                obj_loss = F.cross_entropy(obj_pred, objs)
                attr_loss = F.cross_entropy(attr_pred, attrs)

            loss = obj_loss + attr_loss

        return loss, None


    def val_forward(self, x):
        img_attr, img_obj = x[0], x[1]

        attr_feats1 = self.attr_head1(img_attr)
        obj_feats1 = self.obj_head1(img_obj)
        attr_feats2 = self.attr_head2(self.cross_attention_logit1(attr_feats1, obj_feats1))
        obj_feats2 = self.obj_head2(self.cross_attention_logit2(obj_feats1, attr_feats1))
        ############################################################################
        attr_v = self.cross_attention1(attr_feats2, obj_feats2)
        obj_v = self.cross_attention2(obj_feats2, attr_feats2)
        attr_pred = self.attr_clf(attr_v)
        obj_pred = self.obj_clf(obj_v)
        ############################################################################
        attr_pred = F.softmax(attr_pred, dim=1)
        obj_pred = F.softmax(obj_pred, dim=1)
        score = torch.bmm(attr_pred.unsqueeze(2), obj_pred.unsqueeze(1)).view(attr_pred.shape[0],-1)

        scores = {}
        for itr, (attr, obj) in enumerate(self.dset.pairs):
            attr_id, obj_id = self.dset.attr2idx[attr], self.dset.obj2idx[obj]
            idx = obj_id + attr_id * len(self.dset.objs)
            scores[(attr, obj)] = score[:, idx]

        return score, scores, attr_pred, obj_pred

    def forward(self, x, step):
        if self.training:
            loss, pred = self.train_forward(x, step)
            return loss, pred
        else:
            with torch.no_grad():
                loss, pred, attr_pred, obj_pred = self.val_forward(x)
            return loss, pred, attr_pred, obj_pred



class cross_attention(nn.Module):

    def __init__(self, k_size=7):
        super(cross_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x, y):
        y1 = y.unsqueeze(-1).transpose(-1, -2)
        y2 = self.conv(y1)
        y3 = y2.transpose(-1, -2).squeeze(-1)
        y3 = F.softmax(y3, dim=1)

        return x*y3.expand_as(x) + x  #with residual connection




def freeze(layer):
    for param in layer.parameters():
        param.requires_grad = False

def active(layer):
    for param in layer.parameters():
        param.requires_grad = True



