import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .common_kgsp import MLP
from .gcn import GCN, GCNII
from .word_embedding import load_word_embeddings
import scipy.sparse as sp
from .compcos import compute_cosine_similarity


def adj_to_edges(adj):
    # Adj sparse matrix to list of edges
    rows, cols = np.nonzero(adj)
    edges = list(zip(rows.tolist(), cols.tolist()))
    return edges


def edges_to_adj(edges, n):
    # List of edges to Adj sparse matrix
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype='float32')
    return adj


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GraphFull(nn.Module):
    def __init__(self, dset, args):
        super(GraphFull, self).__init__()
        self.args = args
        self.dset = dset

        self.attrs = dset.attrs
        self.objs = dset.objs

        self.pairs = dset.pairs
        self.pair2idx = dset.all_pair2idx

        self.val_forward = self.val_forward_dotpr
        self.train_forward = self.train_forward_normal

        self.feasibility_adjacency = args.feasibility_adjacency
        self.cosloss = args.cosine_classifier

        self.known_pairs = dset.train_pairs
        seen_pair_set = set(self.known_pairs)
        mask = [1 if pair in seen_pair_set else 0 for pair in self.pairs]
        self.seen_mask = torch.BoolTensor(mask).to(device) * 1.

        self.feasibility_scores = {}
        self.feasibility_margins = -(1 - self.seen_mask).float()
        self.init_feasibility_scores()

        self.epoch_max_margin = self.args.epoch_max_margin
        self.scale = self.args.cosine_scale
        self.cosine_margin_factor = -args.margin

        # Intsantiate attribute-object relations, needed just to evaluate mined pairs
        self.obj_by_attrs_train = {k: [] for k in self.attrs}
        for (a, o) in self.known_pairs:
            self.obj_by_attrs_train[a].append(o)

        # Intanstiate attribute-object relations, needed just to evaluate mined pairs
        self.attrs_by_obj_train = {k: [] for k in self.objs}
        for (a, o) in self.known_pairs:
            self.attrs_by_obj_train[o].append(a)

        # Image Embedder
        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(self.pairs)


        if self.args.train_only:
            train_idx = []
            for current in dset.train_pairs:
                train_idx.append(dset.all_pair2idx[current] + self.num_attrs + self.num_objs)
            self.train_idx = torch.LongTensor(train_idx).to(device)

        self.args.fc_emb = self.args.fc_emb.split(',')
        layers = []
        for a in self.args.fc_emb:
            a = int(a)
            layers.append(a)

        if args.nlayers:
            self.image_embedder = MLP(dset.feat_dim, args.emb_dim, num_layers=args.nlayers, dropout=self.args.dropout,
                                      norm=self.args.norm, layers=layers, relu=True)

        all_words = list(self.dset.attrs) + list(self.dset.objs)
        self.displacement = len(all_words)

        self.obj_to_idx = {word: idx for idx, word in enumerate(self.dset.objs)}
        self.attr_to_idx = {word: idx for idx, word in enumerate(self.dset.attrs)}

        if args.graph_init is not None:
            path = args.graph_init
            graph = torch.load(path)
            embeddings = graph['embeddings'].to(device)
            adj = graph['adj']
            self.embeddings = embeddings
        else:
            embeddings = self.init_embeddings(all_words).to(device)
            adj = self.adj_from_pairs()
            self.embeddings = embeddings

        hidden_layers = self.args.gr_emb
        if args.gcn_type == 'gcn':
            self.gcn = GCN(adj, self.embeddings.shape[1], args.emb_dim, hidden_layers)
        else:
            self.gcn = GCNII(adj, self.embeddings.shape[1], args.emb_dim, args.hidden_dim, args.gcn_nlayers, lamda=0.5,
                             alpha=0.1, variant=False)



    def init_feasibility_scores(self):
        if self.feasibility_adjacency and self.dset.open_world:
            for idx, p in enumerate(self.pairs):
                self.feasibility_scores[p] = 1. if p in self.dset.train_pairs else 0.
                self.feasibility_margins[idx] =  0. if p in self.dset.train_pairs else -10.
        else:
            for idx, p in enumerate(self.pairs):
                self.feasibility_scores[p] = 1.
                self.feasibility_margins[idx] =  0.


    def init_embeddings(self, all_words):

        def get_compositional_embeddings(embeddings, pairs):
            # Getting compositional embeddings from base embeddings
            composition_embeds = []
            for (attr, obj) in pairs:
                attr_embed = embeddings[self.attr_to_idx[attr]]
                obj_embed = embeddings[self.obj_to_idx[obj]+self.num_attrs]
                composed_embed = (attr_embed + obj_embed) / 2
                composition_embeds.append(composed_embed)
            composition_embeds = torch.stack(composition_embeds)
            return composition_embeds

        # init with word embeddings
        embeddings = load_word_embeddings(self.args.emb_init, all_words)

        composition_embeds = get_compositional_embeddings(embeddings, self.pairs)
        full_embeddings = torch.cat([embeddings, composition_embeds], dim=0)

        return full_embeddings


    def update_adj(self, epochs=0.):
        self.compute_feasibility(epochs)
        adj = self.adj_from_pairs()
        self.gcn.update_adj(adj)

    def update_dict(self, wdict, row, col, data):
        wdict['row'].append(row)
        wdict['col'].append(col)
        wdict['data'].append(data)


    def adj_from_pairs(self):

        def edges_from_pairs(pairs):
            weight_dict = {'data': [], 'row': [], 'col': []}

            for i in range(self.displacement):
                self.update_dict(weight_dict, i, i, 1.)

            for idx, (attr, obj) in enumerate(pairs):
                attr_idx, obj_idx = self.attr_to_idx[attr], self.obj_to_idx[obj] + self.num_attrs

                weight = self.feasibility_scores[(attr, obj)] if self.feasibility_adjacency else 1.

                self.update_dict(weight_dict, attr_idx, obj_idx, weight)
                self.update_dict(weight_dict, obj_idx, attr_idx, weight)

                node_id = idx + self.displacement
                self.update_dict(weight_dict, node_id, node_id, 1.)

                self.update_dict(weight_dict, node_id, attr_idx, weight)
                self.update_dict(weight_dict, node_id, obj_idx, weight)

                self.update_dict(weight_dict, attr_idx, node_id, 1.)
                self.update_dict(weight_dict, obj_idx, node_id, 1.)

            return weight_dict

        edges = edges_from_pairs(self.pairs)
        adj = sp.csr_matrix((edges['data'], (edges['row'], edges['col'])),
                            shape=(len(self.pairs) + self.displacement, len(self.pairs) + self.displacement))

        return adj

    def get_pair_scores_objs(self, attr, obj, obj_embedding_sim):
        score = -1.
        for o in self.objs:
            if o != obj and attr in self.attrs_by_obj_train[o]:
                temp_score = obj_embedding_sim[(obj, o)]
                if temp_score > score:
                    score = temp_score
        return score

    def get_pair_scores_attrs(self, attr, obj, attr_embedding_sim):
        score = -1.
        for a in self.attrs:
            if a != attr and obj in self.obj_by_attrs_train[a]:
                temp_score = attr_embedding_sim[(attr, a)]
                if temp_score > score:
                    score = temp_score
        return score


    def compute_feasibility(self,epoch):
        self.gcn.eval()
        embeddings = self.gcn(self.embeddings).detach()

        if self.training:
            self.gcn.train()
        obj_embeddings = embeddings[len(self.attrs):self.displacement]
        obj_embedding_sim = compute_cosine_similarity(self.objs, obj_embeddings,
                                                      return_dict=True)
        attr_embeddings = embeddings[:len(self.attrs)]
        attr_embedding_sim = compute_cosine_similarity(self.attrs, attr_embeddings,
                                                       return_dict=True)

        for (a,o) in self.pairs:
                idx = self.pair2idx[(a, o)]
                if (a, o) not in self.known_pairs:
                    score_obj = self.get_pair_scores_objs(a, o, obj_embedding_sim)
                    score_attr = self.get_pair_scores_attrs(a, o, attr_embedding_sim)
                    score = (score_obj + score_attr) / 2
                    self.feasibility_scores[(a,o)] = max(0.,score)
                    self.feasibility_margins[idx] = score
                else:
                    self.feasibility_scores[(a,o)] = 1.
                    self.feasibility_margins[idx] = 0.
        self.feasibility_margins *= min(1., epoch / self.epoch_max_margin)*self.cosine_margin_factor



    def train_forward_normal(self, x):
        img, attrs, objs, pairs, mask = x[0], x[1], x[2], x[3], x[4]

        if self.args.nlayers:
            img_feats = self.image_embedder(img)
        else:
            img_feats = (img)

        if self.cosloss:
            img_feats = F.normalize(img_feats, dim=1)

        current_embeddings = self.gcn(self.embeddings)

        if self.args.train_only:
            pair_embed = current_embeddings[self.train_idx]
        else:
            pair_embed = current_embeddings[
                         self.num_attrs + self.num_objs:self.num_attrs + self.num_objs + self.num_pairs, :]

        pair_embed = pair_embed.permute(1, 0)
        pair_pred = torch.matmul(img_feats, pair_embed)

        if self.args.partial == True:
            pair_pred = F.softmax(self.scale * pair_pred, dim=-1)
            reshaped_pred = pair_pred.view(-1, len(self.dset.attrs), len(self.dset.objs))
            obj_pred = reshaped_pred.sum(-2)
            attr_pred = reshaped_pred.sum(-1)
            nl = nn.NLLLoss()
            attr_loss = nl(torch.log(attr_pred[mask == 0, :]), attrs[mask == 0])
            obj_loss = nl(torch.log(obj_pred[mask == 1, :]), objs[mask == 1])
            loss = obj_loss + attr_loss

        if self.cosloss:
            if self.dset.open_world:
                pair_pred = (pair_pred + self.feasibility_margins) * self.scale
            else:
                pair_pred = pair_pred * self.scale

            loss = F.cross_entropy(pair_pred, pairs)

        return loss, None

    def val_forward_dotpr(self, x):
        img = x[0]

        if self.args.nlayers:
            img_feats = self.image_embedder(img)
        else:
            img_feats = (img)

        if self.cosloss:
            img_feats = F.normalize(img_feats, dim=1)

        current_embedddings = self.gcn(self.embeddings)

        pair_embeds = current_embedddings[
                      self.num_attrs + self.num_objs:self.num_attrs + self.num_objs + self.num_pairs, :].permute(1, 0)

        score = torch.matmul(img_feats, pair_embeds)

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]

        return score, scores


    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
            return loss, pred
        else:
            with torch.no_grad():
                fast_pred, pred = self.val_forward(x)
            return fast_pred, pred
