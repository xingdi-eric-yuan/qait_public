import torch
import math
import h5py
import numpy as np
import torch.nn.functional as F


def compute_mask(x):
    mask = torch.ne(x, 0).float()
    if x.is_cuda:
        mask = mask.cuda()
    return mask


def masked_softmax(x, m=None, axis=-1):
    '''
    Softmax with mask (optional)
    '''
    x = torch.clamp(x, min=-15.0, max=15.0)
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-6)
    return softmax


def masked_mean(x, m=None, dim=-1):
    """
        mean pooling when there're paddings
        input:  tensor: batch x time x h
                mask:   batch x time
        output: tensor: batch x h
    """
    if m is None:
        return torch.mean(x, dim=dim)
    mask_sum = torch.sum(m, dim=-1)  # batch
    res = torch.sum(x, dim=1)  # batch x h
    res = res / (mask_sum.unsqueeze(-1) + 1e-6)
    return res


def to_one_hot(y_true, n_classes):
    y_onehot = torch.FloatTensor(y_true.size(0), n_classes)
    if y_true.is_cuda:
        y_onehot = y_onehot.cuda()
    y_onehot.zero_()
    y_onehot.scatter_(1, y_true, 1)
    return y_onehot


def NegativeLogLoss(y_pred, y_true):
    """
    Shape:
        - y_pred:    batch x time
        - y_true:    batch
    """
    y_true_onehot = to_one_hot(y_true.unsqueeze(-1), y_pred.size(1))
    P = y_true_onehot.squeeze(-1) * y_pred  # batch x time
    P = torch.sum(P, dim=1)  # batch
    gt_zero = torch.gt(P, 0.0).float()  # batch
    epsilon = torch.le(P, 0.0).float() * 1e-8  # batch
    log_P = torch.log(P + epsilon) * gt_zero  # batch
    output = -log_P  # batch
    return output


def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    length = x.size(1)
    channels = x.size(2)
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)

    signal = signal.cuda() if x.is_cuda else signal
    return x + signal

def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales)-1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    m = torch.nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


class LayerNorm(torch.nn.Module):

    def __init__(self, input_dim):
        super(LayerNorm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(input_dim))
        self.beta = torch.nn.Parameter(torch.zeros(input_dim))
        self.eps = 1e-6

    def forward(self, x, mask):
        # x:        nbatch x hidden
        # mask:     nbatch
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True) + self.eps)
        output = self.gamma * (x - mean) / (std + self.eps) + self.beta
        return output * mask.unsqueeze(1)


class H5EmbeddingManager(object):
    def __init__(self, h5_path):
        f = h5py.File(h5_path, 'r')
        self.W = np.array(f['embedding'])
        print("embedding data type=%s, shape=%s" % (type(self.W), self.W.shape))
        self.id2word = f['words_flatten'][0].split('\n')
        self.word2id = dict(zip(self.id2word, range(len(self.id2word))))

    def __getitem__(self, item):
        item_type = type(item)
        if item_type is str:
            index = self.word2id[item]
            embs = self.W[index]
            return embs
        else:
            raise RuntimeError("don't support type: %s" % type(item))

    def word_embedding_initialize(self, words_list, dim_size=300, scale=0.1, oov_init='random'):
        shape = (len(words_list), dim_size)
        np.random.seed(42)
        if 'zero' == oov_init:
            W2V = np.zeros(shape, dtype='float32')
        elif 'one' == oov_init:
            W2V = np.ones(shape, dtype='float32')
        else:
            W2V = np.random.uniform(low=-scale, high=scale, size=shape).astype('float32')
        W2V[0, :] = 0
        in_vocab = np.ones(shape[0], dtype=np.bool)
        word_ids = []
        for i, word in enumerate(words_list):
            if word in self.word2id:
                word_ids.append(self.word2id[word])
            else:
                in_vocab[i] = False
        W2V[in_vocab] = self.W[np.array(word_ids, dtype='int32')][:, :dim_size]
        return W2V


class Embedding(torch.nn.Module):
    '''
    inputs: x:          batch x ...
    outputs:embedding:  batch x ... x emb
            mask:       batch x ...
    '''

    def __init__(self, embedding_size, vocab_size, dropout_rate=0.0, trainable=True, id2word=None,
                 embedding_oov_init='random', load_pretrained=False, pretrained_embedding_path=None):
        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.id2word = id2word
        self.dropout_rate = dropout_rate
        self.load_pretrained = load_pretrained
        self.embedding_oov_init = embedding_oov_init
        self.pretrained_embedding_path = pretrained_embedding_path
        self.trainable = trainable
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        self.init_weights()

    def init_weights(self):
        init_embedding_matrix = self.embedding_init()
        if self.embedding_layer.weight.is_cuda:
            init_embedding_matrix = init_embedding_matrix.cuda()
        self.embedding_layer.weight = torch.nn.Parameter(init_embedding_matrix)
        if not self.trainable:
            self.embedding_layer.weight.requires_grad = False

    def embedding_init(self):
        # Embeddings
        if self.load_pretrained is False:
            word_embedding_init = np.random.uniform(low=-0.05, high=0.05, size=(self.vocab_size, self.embedding_size))
            word_embedding_init[0, :] = 0
        else:
            embedding_initr = H5EmbeddingManager(self.pretrained_embedding_path)
            word_embedding_init = embedding_initr.word_embedding_initialize(self.id2word,
                                                                            dim_size=self.embedding_size,
                                                                            oov_init=self.embedding_oov_init)
            del embedding_initr
        word_embedding_init = torch.from_numpy(word_embedding_init).float()
        return word_embedding_init

    def compute_mask(self, x):
        mask = torch.ne(x, 0).float()
        if x.is_cuda:
            mask = mask.cuda()
        return mask

    def forward(self, x):
        embeddings = self.embedding_layer(x)  # batch x time x emb
        embeddings = F.dropout(embeddings, p=self.dropout_rate, training=self.training)
        mask = self.compute_mask(x)  # batch x time
        return embeddings, mask


class NoisyLinear(torch.nn.Module):
    # Factorised NoisyLinear layer with bias
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = torch.nn.Parameter(torch.empty(out_features))
        self.bias_sigma = torch.nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()
        self._zero_noise = False

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def zero_noise(self):
        self._zero_noise = True

    def forward(self, input):
        if self.training:
            if self._zero_noise is True:
                return F.linear(input, self.weight_mu, self.bias_mu)
            else:
                return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class DepthwiseSeparableConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = torch.nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        self.pointwise_conv = torch.nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        x = x.transpose(1,2)
        res = torch.relu(self.pointwise_conv(self.depthwise_conv(x)))
        res = res.transpose(1,2)
        return res


class SelfAttention(torch.nn.Module):
    def __init__(self, block_hidden_dim, n_head, dropout):
        super().__init__()
        self.block_hidden_dim = block_hidden_dim
        self.n_head = n_head
        self.dropout = dropout
        self.key_linear = torch.nn.Linear(block_hidden_dim, block_hidden_dim, bias=False)
        self.value_linear = torch.nn.Linear(block_hidden_dim, block_hidden_dim, bias=False)
        self.query_linear = torch.nn.Linear(block_hidden_dim, block_hidden_dim, bias=False)
        bias = torch.empty(1)
        torch.nn.init.constant_(bias, 0)
        self.bias = torch.nn.Parameter(bias)

    def forward(self, queries, query_mask, keys, values):

        query = self.query_linear(queries)
        key = self.key_linear(keys)
        value = self.value_linear(values)
        Q = self.split_last_dim(query, self.n_head)
        K = self.split_last_dim(key, self.n_head)
        V = self.split_last_dim(value, self.n_head)
        
        assert self.block_hidden_dim % self.n_head == 0
        key_depth_per_head = self.block_hidden_dim // self.n_head
        Q *= key_depth_per_head**-0.5
        x = self.dot_product_attention(Q, K, V, mask=query_mask)
        return self.combine_last_two_dim(x.permute(0, 2, 1, 3))

    def dot_product_attention(self, q, k ,v, bias=False, mask=None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            logits += self.bias
        if mask is not None:
            # shapes = [x if x != None else -1 for x in list(logits.size())]
            # mask = mask.view(shapes[0], 1, 1, shapes[-1])
            mask = mask.unsqueeze(1)
        weights = masked_softmax(logits, mask, axis=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret


class EncoderBlock(torch.nn.Module):
    def __init__(self, conv_num, ch_num, k, block_hidden_dim, n_head, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.FFN_1 = torch.nn.Linear(ch_num, ch_num)
        self.FFN_2 = torch.nn.Linear(ch_num, ch_num)
        self.norm_C = torch.nn.ModuleList([torch.nn.LayerNorm(block_hidden_dim) for _ in range(conv_num)])
        self.norm_1 = torch.nn.LayerNorm(block_hidden_dim)
        self.norm_2 = torch.nn.LayerNorm(block_hidden_dim)
        self.conv_num = conv_num

    def forward(self, x, mask, self_att_mask, l, blks):
        total_layers = (self.conv_num + 2) * blks
        # conv layers
        out = PosEncoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out)
            if (i) % 2 == 0:
                out = F.dropout(out, p=self.dropout, training=self.training)
            out = conv(out)
            out = out * mask.unsqueeze(-1)
            out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
            l += 1
        res = out
        out = self.norm_1(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # self attention
        out = self.self_att(out, self_att_mask, out, out)
        out = out * mask.unsqueeze(-1)
        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        l += 1
        res = out
        out = self.norm_2(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # fully connected layers
        out = self.FFN_1(out)
        out = torch.relu(out)
        out = self.FFN_2(out)
        out = out * mask.unsqueeze(-1)
        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        l += 1
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class CQAttention(torch.nn.Module):
    def __init__(self, block_hidden_dim, dropout=0):
        super().__init__()
        self.dropout = dropout
        w4C = torch.empty(block_hidden_dim, 1)
        w4Q = torch.empty(block_hidden_dim, 1)
        w4mlu = torch.empty(1, 1, block_hidden_dim)
        torch.nn.init.xavier_uniform_(w4C)
        torch.nn.init.xavier_uniform_(w4Q)
        torch.nn.init.xavier_uniform_(w4mlu)
        self.w4C = torch.nn.Parameter(w4C)
        self.w4Q = torch.nn.Parameter(w4Q)
        self.w4mlu = torch.nn.Parameter(w4mlu)

        bias = torch.empty(1)
        torch.nn.init.constant_(bias, 0)
        self.bias = torch.nn.Parameter(bias)

    def forward(self, C, Q, Cmask, Qmask):
        S = self.trilinear_for_attention(C, Q)
        Cmask = Cmask.unsqueeze(-1)
        Qmask = Qmask.unsqueeze(1)
        S1 = masked_softmax(S, Qmask, axis=2)
        S2 = masked_softmax(S, Cmask, axis=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out

    def trilinear_for_attention(self, C, Q):
        C = F.dropout(C, p=self.dropout, training=self.training)
        Q = F.dropout(Q, p=self.dropout, training=self.training)
        max_q_len = Q.size(-2)
        max_context_len = C.size(-2)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, max_q_len])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, max_context_len, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res


class AnswerPointer(torch.nn.Module):
    def __init__(self, block_hidden_dim, noisy_net=False):
        super().__init__()
        self.noisy_net = noisy_net
        if self.noisy_net:
            self.w_1 = NoisyLinear(block_hidden_dim * 2, 1)
            self.w_1_advantage = NoisyLinear(block_hidden_dim * 2, block_hidden_dim)
            self.w_2 = NoisyLinear(block_hidden_dim, 1)
        else:
            self.w_1 = torch.nn.Linear(block_hidden_dim * 2, 1)
            self.w_1_advantage = torch.nn.Linear(block_hidden_dim * 2, block_hidden_dim)
            self.w_2 = torch.nn.Linear(block_hidden_dim, 1)

    def forward(self, M1, M2, mask):
        X_concat = torch.cat([M1, M2], dim=-1)
        X = torch.relu(self.w_1(X_concat))
        X_advantage = torch.relu(self.w_1_advantage(X_concat))
        X = X * mask.unsqueeze(-1)
        X = X + X_advantage - X_advantage.mean(-1, keepdim=True)  # combine streams
        X = X * mask.unsqueeze(-1)
        Y = self.w_2(X).squeeze()
        Y = Y * mask
        return Y

    def reset_noise(self):
        if self.noisy_net:
            self.w_1.reset_noise()
            self.w_1_advantage.reset_noise()
            self.w_2.reset_noise()
    
    def zero_noise(self):
        if self.noisy_net:
            self.w_1.zero_noise()
            self.w_1_advantage.zero_noise()
            self.w_2.zero_noise()


class Highway(torch.nn.Module):
    def __init__(self, layer_num, size, dropout=0):
        super().__init__()
        self.n = layer_num
        self.dropout = dropout
        self.linear = torch.nn.ModuleList([torch.nn.Linear(size, size) for _ in range(self.n)])
        self.gate = torch.nn.ModuleList([torch.nn.Linear(size, size) for _ in range(self.n)])

    def forward(self, x):
        #x: shape [batch_size, hidden_size, length]
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=self.dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
        return x


class MergeEmbeddings(torch.nn.Module):
    def __init__(self, block_hidden_dim, word_emb_dim, char_emb_dim, dropout=0):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(char_emb_dim, block_hidden_dim, kernel_size = (1, 5), padding=0, bias=True)
        torch.nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')

        self.linear = torch.nn.Linear(word_emb_dim + block_hidden_dim, block_hidden_dim, bias=False)
        self.high = Highway(2, size=block_hidden_dim, dropout=dropout)

    def forward(self, word_emb, char_emb, mask=None):
        char_emb = char_emb.permute(0, 3, 1, 2)  # batch x emb x time x nchar
        char_emb = self.conv2d(char_emb)  # batch x block_hidden_dim x time x nchar-5+1
        if mask is not None:
            char_emb = char_emb * mask.unsqueeze(1).unsqueeze(-1)
        char_emb = F.relu(char_emb)  # batch x block_hidden_dim x time x nchar-5+1
        char_emb, _ = torch.max(char_emb, dim=3)  # batch x emb x time
        char_emb = char_emb.permute(0, 2, 1)  # batch x time x emb
        emb = torch.cat([char_emb, word_emb], dim=2)
        emb = self.linear(emb)
        emb = self.high(emb)
        if mask is not None:
            emb = emb * mask.unsqueeze(-1)
        return emb
