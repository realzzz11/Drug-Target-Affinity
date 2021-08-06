import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import pickle
import math
import numpy as np
import torch.optim as optim
from utils import params, batch2tensor, regression_scores, init_embed
from sklearn.model_selection import KFold
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class GATLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(3), self.alpha)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, Wh)

        return F.elu(h_prime) if self.concat else h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        b = Wh.size()[0]
        N = Wh.size()[1]

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat_interleave(N, dim=0).view(b, N*N, self.out_features)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)

        return all_combinations_matrix.view(b, N, N, 2 * self.out_features)


class BiDACPI(nn.Module):
    def __init__(self, task, init_atom, init_amino, params):
        super(BiDACPI, self).__init__()

        comp_dim, prot_dim, gat_dim, num_head, dropout, alpha, window, layer_cnn, latent_dim, layer_out = \
            params.comp_dim, params.prot_dim, params.gat_dim, params.num_head, params.dropout, params.alpha,\
            params.window, params.layer_cnn, params.latent_dim, params.layer_out

        self.embed_atom = nn.Embedding(len(init_atom), 82, padding_idx=0)
        self.embed_atom.weight = nn.Parameter(init_atom)
        self.embed_atom.weight.requires_grad = False

        self.embed_amino = nn.Embedding(len(init_amino), 20, padding_idx=0)
        self.embed_amino.weight = nn.Parameter(init_amino)
        self.embed_amino.weight.requires_grad = False

        self.embedding_layer_atom = nn.Linear(82, comp_dim)

        self.dropout = dropout
        self.alpha = alpha
        self.layer_cnn = layer_cnn
        self.layer_out = layer_out

        self.gat_layers = [GATLayer(comp_dim, gat_dim, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(num_head)]
        for i, layer in enumerate(self.gat_layers):
            self.add_module('gat_layer_{}'.format(i), layer)
        self.gat_out = GATLayer(gat_dim * num_head, comp_dim, dropout=dropout, alpha=alpha, concat=False)
        self.W_comp = nn.Linear(comp_dim, latent_dim)

        self.conv_first = nn.Conv1d(20, prot_dim, kernel_size=2 * window + 1, padding=window)
        self.cnn_layers = nn.ModuleList([])
        for i in range(self.layer_cnn):
            self.cnn_layers.append(nn.Conv1d(prot_dim, prot_dim, kernel_size=2 * window + 1, padding=window))
        self.W_prot = nn.Linear(prot_dim, latent_dim)

        self.W_attention = nn.Linear(latent_dim, latent_dim)
        self.att_atoms2prot = nn.Linear(2 * latent_dim, 1)
        self.att_amino2comp = nn.Linear(2 * latent_dim, 1)
        self.W_out = nn.ModuleList([nn.Linear(2 * latent_dim, 2 * latent_dim) for _ in range(layer_out)])

        if task == 'interaction':
            self.predict = nn.Linear(2 * latent_dim, 2)
        elif task == 'affinity':
            self.predict = nn.Linear(2 * latent_dim, 1)

    def comp_gat(self, atoms, atoms_mask, adj):
        atoms_ebd = self.embed_atom(atoms)
        atoms_vector = F.leaky_relu(self.embedding_layer_atom(atoms_ebd), 0.1)
        atoms_multi_head = torch.cat([gat(atoms_vector, adj) for gat in self.gat_layers], dim=2)
        atoms_multi_head = F.dropout(atoms_multi_head, self.dropout, training=self.training)
        atoms_vector = F.elu(self.gat_out(atoms_multi_head, adj))
        atoms_vector = F.leaky_relu(self.W_comp(atoms_vector), self.alpha)
        return atoms_vector

    def prot_cnn(self, amino, amino_mask):
        amino_ebd = self.embed_amino(amino)
        amino_ebd = amino_ebd.transpose(1, 2)
        amino_vector = F.leaky_relu(self.conv_first(amino_ebd), 0.1)
        for i in range(self.layer_cnn):
            amino_vector = F.leaky_relu(self.cnn_layers[i](amino_vector), 0.1)
        return amino_vector.transpose(1, 2)

    def bidirectional_attention_prediction(self, atoms_vector, atoms_mask, amino_vector, amino_mask):
        b = atoms_vector.shape[0]

        atoms_vector = F.leaky_relu(self.W_attention(atoms_vector), self.alpha)
        amino_vector = F.leaky_relu(self.W_attention(amino_vector), self.alpha)

        prot_vector = torch.sum(amino_vector * amino_mask.view(b, -1, 1), dim=1) / torch.sum(amino_mask, dim=1, keepdim=True)
        prot_rep = torch.unsqueeze(prot_vector, 1).repeat_interleave(atoms_vector.shape[1], dim=1)
        Wh_atoms2prot = torch.cat([atoms_vector, prot_rep], dim=2)

        atoms_attention = torch.tanh(self.att_atoms2prot(Wh_atoms2prot))
        atoms_vector = atoms_vector * atoms_attention
        comp_vector = torch.sum(atoms_vector * atoms_mask.view(b, -1, 1), dim=1) / torch.sum(atoms_mask, dim=1, keepdim=True)

        comp_rep = torch.unsqueeze(comp_vector, 1).repeat_interleave(amino_vector.shape[1], dim=1)
        Wh_amino2comp = torch.cat([amino_vector, comp_rep], dim=2)

        amino_attention = torch.tanh((self.att_amino2comp(Wh_amino2comp)))
        amino_vector = amino_vector * amino_attention
        prot_vector = torch.sum(amino_vector * amino_mask.view(b, -1, 1), dim=1) / torch.sum(amino_mask, dim=1, keepdim=True)

        comp_prot_vector = torch.cat((comp_vector, prot_vector), dim=1)
        for i in range(self.layer_out):
            comp_prot_vector = F.leaky_relu(self.W_out[i](comp_prot_vector), self.alpha)
        return self.predict(comp_prot_vector)

    def forward(self, atoms, atoms_mask, adjacency, amino, amino_mask):
        import time
        import numpy as np
        st = time.time()
        atoms_vector = self.comp_gat(atoms, atoms_mask, adjacency)
        t1 = time.time() - st
        amino_vector = self.prot_cnn(amino, amino_mask)
        t2 = time.time() - st - t1
        prediction = self.bidirectional_attention_prediction(atoms_vector, atoms_mask, amino_vector, amino_mask)
        t3 = time.time() - st - t1 - t2
        t = np.array([t1, t2, t3])
        # print(t / t.sum())
        return prediction


def print2file(buf, outFile, p=False):
    if p:
        print(buf)
    outfd = open(outFile, 'a+')
    outfd.write(buf + '\n')
    outfd.close()


def train_eval(model, task, train_data, valid_data, test_data, device, params):
    criterion = F.mse_loss if task == 'affinity' else F.cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    idx = np.arange(len(train_data[0]))
    batch_size = params.batch_size
    # min_loss = 1000
    for epoch in range(params.num_epochs):
        print2file('epoch:{}'.format(epoch), 'bi-model.txt', True)
        np.random.shuffle(idx)
        for i in range(math.ceil(len(train_data[0]) / batch_size)):
            batch_data = [train_data[di][idx[i * batch_size: (i + 1) * batch_size]] for di in range(4)]
            atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, label = batch2tensor(batch_data, device)
            pred = model(atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask)
            loss = criterion(pred.float(), label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write('\repoch:{}, batch:{}/{}, loss:{}'
                             .format(epoch, i, math.ceil(len(train_data[0])/batch_size)-1, float(loss.data)))
            sys.stdout.flush()

        rmse_train, pearson_train, spearman_train = test(model, task, train_data, batch_size, device)
        info = '\nTrain rmse:{}, pearson:{}, spearman:{}'.format(rmse_train, pearson_train, spearman_train)
        print2file(info, 'bi-model.txt', True)

        rmse_valid, pearson_valid, spearman_valid = test(model, task, valid_data, batch_size, device)
        info = 'Valid rmse:{}, pearson:{}, spearman:{}'.format(rmse_valid, pearson_valid, spearman_valid)
        print2file(info, 'bi-model.txt', True)

        rmse_test, pearson_test, spearman_test = test(model, task, test_data, batch_size, device)
        info = 'Test rmse:{}, pearson:{}, spearman:{}'.format(rmse_test, pearson_test, spearman_test)
        print2file(info, 'bi-model.txt', True)

        scheduler.step()


def test(model, task, test_data, batch_size, device):
    model.eval()
    predictions = []
    labels = []
    for i in range(math.ceil(len(test_data[0]) / batch_size)):
        batch_data = [test_data[di][i * batch_size: (i + 1) * batch_size] for di in range(4)]
        atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, label = batch2tensor(batch_data, device)
        with torch.no_grad():
            pred = model(atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask)
        if task == 'affinity':
            predictions += pred.cpu().detach().numpy().reshape(-1).tolist()
            labels += label.cpu().numpy().reshape(-1).tolist()
        else:
            pass
    predictions = np.array(predictions)
    labels = np.array(labels)
    if task == 'affinity':
        rmse_value, pearson_value, spearman_value = regression_scores(labels, predictions)
        return round(rmse_value, 6), round(pearson_value, 6), round(spearman_value, 6)
    else:
        pass


if __name__ == '__main__':
    task = 'affinity'
    DATASET = 'BindingDB'
    measure = 'IC50'  # 'Ki', 'IC50', 'Kd', 'EC50'
    target_class = 'test'  # 'GPCR', 'ER', 'channel', 'kinase'

    dir_input = ('../datasets/' + DATASET + '/' + measure + '_' + target_class + '/')

    device = torch.device('cuda') if params.mode == 'gpu' and torch.cuda.is_available() else torch.device('cpu')
    print('The code run on the', device)

    print('Load data...')
    compounds = np.load(dir_input + 'compounds.npy', allow_pickle=True)
    adjacencies = np.load(dir_input + 'adjacencies.npy', allow_pickle=True)
    fingerprint = np.load(dir_input + 'fingerprint.npy', allow_pickle=True)
    proteins = np.load(dir_input + 'proteins.npy', allow_pickle=True)
    interactions = np.load(dir_input + 'interactions.npy', allow_pickle=True)
    dataset = [compounds, adjacencies, proteins, interactions]

    init_atom, init_amino = init_embed(dir_input)

    print('training...')
    kf = KFold(5, shuffle=True)
    for train_valid_idx, test_idx in kf.split(range(len(dataset[0]))):
        model = BiDACPI(task, init_atom, init_amino, params)
        model.to(device)

        valid_idx = np.random.choice(train_valid_idx, int(len(train_valid_idx) * 0.125), replace=False)
        train_idx = list(set(train_valid_idx) - set(valid_idx))
        print('train num:', len(train_idx), 'valid num:', len(valid_idx), 'test num:', len(test_idx))

        train_data = [dataset[i][train_idx] for i in range(4)]
        valid_data = [dataset[i][valid_idx] for i in range(4)]
        test_data = [dataset[i][test_idx] for i in range(4)]

        train_eval(model, task, train_data, valid_data, test_data, device, params)