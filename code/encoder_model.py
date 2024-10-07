import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, Wh)

        return F.elu(h_prime) if self.concat else h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        b = Wh.size()[0]
        N = Wh.size()[1]

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat_interleave(N, dim=0).view(b, N*N, self.out_features)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)

        return all_combinations_matrix.view(b, N, N, 2 * self.out_features)

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size , dropout, device):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"
        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.do = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)
        for i, conv in enumerate(self.convs):
            conved = conv(self.do(conv_input))
            conved = F.glu(conved, dim=1)
            conved = (conved + conv_input) * self.scale
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        conved = self.ln(conved)
        return conved

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.do(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        x = x.permute(0, 2, 1)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        trg = self.ln(trg + self.do(self.pf(trg)))
        return trg

class Decoder(nn.Module):
    def __init__(self, embed_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super(Decoder, self).__init__()
        self.ft = nn.Linear(embed_dim, hid_dim)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])
        self.do = nn.Dropout(dropout)
        self.device = device

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.do(self.ft(trg))
        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)
        return trg #[bs, seq_len, hid_dim]

class BACPI(nn.Module):
    def __init__(self, task, n_atom, n_amino, params, device):
        super(BACPI, self).__init__()

        comp_dim, prot_dim, gat_dim, num_head, dropout, alpha, window, layer_cnn, latent_dim, layer_out = \
            params.comp_dim, params.prot_dim, params.gat_dim, params.num_head, params.dropout, params.alpha,\
            params.window, params.layer_cnn, params.latent_dim, params.layer_out

        self.device = device
        self.embedding_layer_atom = nn.Embedding(n_atom+1, comp_dim)
        self.embedding_layer_amino = nn.Embedding(n_amino+1, prot_dim)

        self.dropout = dropout
        self.alpha = alpha
        self.layer_cnn = layer_cnn
        self.layer_out = layer_out

        # GAT layers for compound representation
        self.gat_layers = [GATLayer(comp_dim, gat_dim, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(num_head)]
        for i, layer in enumerate(self.gat_layers):
            self.add_module('gat_layer_{}'.format(i), layer)
        self.gat_out = GATLayer(gat_dim * num_head, comp_dim, dropout=dropout, alpha=alpha, concat=False)
        self.W_comp = nn.Linear(comp_dim, latent_dim)

        # CNN layers for protein representation
        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2*window+1,
                                                    stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_prot = nn.Linear(prot_dim, latent_dim)

        # Bidirectional encoder-decoder layers
        self.compound_encoder = Encoder(comp_dim, latent_dim, n_layers=2, kernel_size=3, dropout=dropout, device=device)
        self.compound_decoder = Decoder(comp_dim, latent_dim, n_layers=2, n_heads=4, pf_dim=latent_dim*2, 
                                    decoder_layer=DecoderLayer, self_attention=SelfAttention, 
                                    positionwise_feedforward=PositionwiseFeedforward, dropout=dropout, device=device)

        self.protein_encoder = Encoder(prot_dim, latent_dim, n_layers=2, kernel_size=3, dropout=dropout, device=device)
        self.protein_decoder = Decoder(prot_dim, latent_dim, n_layers=2, n_heads=4, pf_dim=latent_dim*2, 
                               decoder_layer=DecoderLayer, self_attention=SelfAttention, 
                               positionwise_feedforward=PositionwiseFeedforward, dropout=dropout, device=device)
        self.fp0 = nn.Parameter(torch.empty(size=(1024, latent_dim)))
        nn.init.xavier_uniform_(self.fp0, gain=1.414)
        self.fp1 = nn.Parameter(torch.empty(size=(latent_dim, latent_dim)))
        nn.init.xavier_uniform_(self.fp1, gain=1.414)

        # Bidirectional attention mechanism
        self.bidat_num = 4
        self.U = nn.ParameterList([nn.Parameter(torch.empty(size=(latent_dim, latent_dim))) for _ in range(self.bidat_num)])
        for i in range(self.bidat_num):
            nn.init.xavier_uniform_(self.U[i], gain=1.414)

        self.transform_c2p = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.transform_p2c = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.bihidden_c = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.bihidden_p = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.biatt_c = nn.ModuleList([nn.Linear(latent_dim * 2, 1) for _ in range(self.bidat_num)])
        self.biatt_p = nn.ModuleList([nn.Linear(latent_dim * 2, 1) for _ in range(self.bidat_num)])

        self.comb_c = nn.Linear(latent_dim * self.bidat_num, latent_dim)
        self.comb_p = nn.Linear(latent_dim * self.bidat_num, latent_dim)

        if task == 'affinity':
            self.output = nn.Linear(latent_dim * latent_dim * 2, 1)
        elif task == 'interaction':
            self.output = nn.Linear(latent_dim * latent_dim * 2, 2)
        else:
            print("Please choose a correct mode!!!")

    def comp_gat(self, atoms, atoms_mask, adj):
        atoms_vector = self.embedding_layer_atom(atoms)
        atoms_multi_head = torch.cat([gat(atoms_vector, adj) for gat in self.gat_layers], dim=2)
        atoms_vector = F.elu(self.gat_out(atoms_multi_head, adj))
        atoms_vector = F.leaky_relu(self.W_comp(atoms_vector), self.alpha)
        return atoms_vector

    def prot_cnn(self, amino, amino_mask):
        amino_vector = self.embedding_layer_amino(amino)
        amino_vector = torch.unsqueeze(amino_vector, 1)
        for i in range(self.layer_cnn):
            amino_vector = F.leaky_relu(self.conv_layers[i](amino_vector), self.alpha)
        amino_vector = torch.squeeze(amino_vector, 1)
        amino_vector = F.leaky_relu(self.W_prot(amino_vector), self.alpha)
        return amino_vector
    def mask_softmax(self, a, mask, dim=-1):
        a_max = torch.max(a, dim, keepdim=True)[0]
        a_exp = torch.exp(a - a_max)
        a_exp = a_exp * mask
        a_softmax = a_exp / (torch.sum(a_exp, dim, keepdim=True) + 1e-6)
        return a_softmax
    
    def encoder_decoder(self, atoms_vector, amino_vector):
        compound_encoded = self.compound_encoder(atoms_vector)
        protein_encoded = self.protein_encoder(amino_vector)
        compound_decoded = self.compound_decoder(compound_encoded, protein_encoded)
        protein_decoded = self.protein_decoder(protein_encoded, compound_encoded)

        return compound_decoded, protein_decoded

    def bidirectional_attention_prediction(self, atoms_vector, atoms_mask, fps, amino_vector, amino_mask):
        b = atoms_vector.shape[0]
        for i in range(self.bidat_num):
            A = torch.tanh(torch.matmul(torch.matmul(atoms_vector, self.U[i]), amino_vector.transpose(1, 2)))
            A_masked = A * torch.matmul(atoms_mask.view(b, -1, 1), amino_mask.view(b, 1, -1))

            # atoms_trans:经过蛋白质信息加权的化合物特征
            atoms_trans = torch.matmul(A_masked, torch.tanh(self.transform_p2c[i](amino_vector))) # 蛋白质特征amino_vector经过线性变换self.transform_p2c[i]，对蛋白质特征进行调整。
            amino_trans = torch.matmul(A_masked.transpose(1, 2), torch.tanh(self.transform_c2p[i](atoms_vector)))

            # atoms_combined_features：通过self.bihidden_c[i]线性变换，得到化合物原始特征和通过蛋白质加权得到的化合物特征
            atoms_combined_features = torch.cat([torch.tanh(self.bihidden_c[i](atoms_vector)), atoms_trans], dim=2)
            amino_combined_features = torch.cat([torch.tanh(self.bihidden_p[i](amino_vector)), amino_trans], dim=2)

            # atoms_attention_weights：化合物的注意力权重
            atoms_attention_weights = self.mask_softmax(self.biatt_c[i](atoms_combined_features).view(b, -1), atoms_mask.view(b, -1)) # self.biatt_c[i](atoms_combined_features)：将每个原子的重要性映射为一个分数 ｜ view(b, -1)：展平
            amino_attention_weights = self.mask_softmax(self.biatt_p[i](amino_combined_features).view(b, -1), amino_mask.view(b, -1))

            # 加权求和
            cf = torch.sum(atoms_vector * atoms_attention_weights.view(b, -1, 1), dim=1)
            pf = torch.sum(amino_vector * amino_attention_weights.view(b, -1, 1), dim=1)

            if i == 0:
                cat_cf = cf
                cat_pf = pf
            # 累加
            else:
                cat_cf = torch.cat([cat_cf.view(b, -1), cf.view(b, -1)], dim=1)
                cat_pf = torch.cat([cat_pf.view(b, -1), pf.view(b, -1)], dim=1)
        
        # cf_final：所有注意力循环加起来的化合物+fps
        cf_final = torch.cat([self.comb_c(cat_cf).view(b, -1), fps.view(b, -1)], dim=1)
        pf_final = self.comb_p(cat_pf)

        # 化合物特征cf_final与蛋白质特征pf_final进行交互
        cf_pf = F.leaky_relu(torch.matmul(cf_final.view(b, -1, 1), pf_final.view(b, 1, -1)).view(b, -1), 0.1)
        return self.output(cf_pf)

    def forward(self, atoms, atoms_mask, adjacency, amino, amino_mask, fps):
        # Compound GAT and Protein CNN
        atoms_vector = self.comp_gat(atoms, atoms_mask, adjacency)
        amino_vector = self.prot_cnn(amino, amino_mask)

        # Global feature transformations
        transformed_fps_1 = F.leaky_relu(torch.matmul(fps, self.fp0), 0.1)
        transformed_fps_2 = F.leaky_relu(torch.matmul(transformed_fps_1, self.fp1), 0.1)

        # Encoder-Decoder process
        atoms_decoded, amino_decoded = self.encoder_decoder(atoms_vector, amino_vector)

        # Bidirectional attention prediction
        prediction = self.bidirectional_attention_prediction(atoms_decoded, atoms_mask, transformed_fps_2, amino_decoded, amino_mask)
        return prediction
