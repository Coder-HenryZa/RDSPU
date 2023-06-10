import torch as th
from torch_geometric.nn import GCNConv, GINConv
import copy
import pickle
from networks import WordTransformer
from networks import PostTransformer
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
from networks.Attention import Attention

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
import sys
sys.path.append('./networks')


class WordLightBert(th.nn.Module):
    def __init__(self, dataname, word_embedding_dim, word_head, word_dff, word_droupout, word_N, word_mask):
        super(WordLightBert, self).__init__()
        self.ws = pickle.load(open('./datasets/' + dataname + '_ws.pkl', 'rb'))
        self.c = copy.deepcopy

        # word
        self.word_embedding_dim = word_embedding_dim
        self.word_head = word_head
        self.word_dff = word_dff
        self.word_droupout = word_droupout
        self.word_N = word_N
        self.word_mask = word_mask
        self.word_embedding = WordTransformer.Embeddings(self.word_embedding_dim, len(self.ws))
        self.word_attention = WordTransformer.MultiHeadedAttention(self.word_head, self.word_embedding_dim)
        self.word_ff = WordTransformer.PositionwiseFeedForward(self.word_embedding_dim, self.word_dff,
                                                               self.word_droupout)
        self.word_layer = WordTransformer.EncoderLayer(self.word_embedding_dim, self.c(self.word_attention),
                                                       self.c(self.word_ff), self.word_droupout)
        self.word_en = WordTransformer.Encoder(self.word_layer, self.word_N)


class PostLightBert(th.nn.Module):
    def __init__(self, dataname, post_embedding_dim, post_head, post_dff, post_droupout, post_N, post_mask):
        super(PostLightBert, self).__init__()
        self.ws = pickle.load(open('./datasets/'+dataname + '_ws.pkl', 'rb'))
        self.c = copy.deepcopy
        # post
        self.post_embedding_dim = post_embedding_dim
        self.post_head = post_head
        self.post_dff = post_dff
        self.post_droupout = post_droupout
        self.post_N = post_N
        self.post_mask = post_mask
        self.post_embedding = PostTransformer.Embeddings(self.post_embedding_dim, len(self.ws))
        self.post_attention = PostTransformer.MultiHeadedAttention(self.post_head, self.post_embedding_dim)
        self.post_ff = PostTransformer.PositionwiseFeedForward(self.post_embedding_dim, self.post_dff,
                                                               self.post_droupout)
        self.post_layer = PostTransformer.EncoderLayer(self.post_embedding_dim, self.c(self.post_attention),
                                                       self.c(self.post_ff), self.post_droupout)
        self.post_en = PostTransformer.Encoder(self.post_layer, self.post_N)


class StatusesEncoder(th.nn.Module):
    def __init__(self, dataname, word_embedding_dim, post_embedding_dim, word_head, post_head, word_dff, post_dff, word_droupout,
                 post_droupout, word_N, post_N, word_mask, post_mask, num_words,
                 num_posts, out_dim, atten_out_dim):
        super(StatusesEncoder, self).__init__()

        self.num_posts = num_posts
        self.num_words = num_words
        self.out_dim = out_dim
        self.wordLightBert = WordLightBert(dataname=dataname,word_embedding_dim=word_embedding_dim, word_head=word_head,
                                           word_dff=word_dff, word_droupout=word_droupout, word_N=word_N,
                                           word_mask=word_mask)
        self.postLightBert = PostLightBert(dataname=dataname,post_embedding_dim=post_embedding_dim, post_head=post_head,
                                           post_dff=post_dff, post_droupout=post_droupout, post_N=post_N,
                                           post_mask=post_mask)
        self.post_linear = th.nn.Linear(self.postLightBert.post_embedding_dim, self.out_dim)

        self.Attention = Attention(init_size=post_embedding_dim,hidden_size=post_embedding_dim,output_size=atten_out_dim)
    def forward(self, data):
        """
        最终输出 （128*128的数据）

        """
        # word transformer
        history_statues = data.history_text  # (batch_size, num_posts, num_words)

        batch_size = history_statues.shape[0]

        x = self.wordLightBert.word_embedding(history_statues)  # (batch_size, num_posts, num_words,embedding_dim)
        x = self.wordLightBert.word_en(x, mask=None)  # (batch_size, num_posts, num_words,embedding_dim)
        # word_注意力最大化池化
        x = x.view(-1, self.num_words, self.wordLightBert.word_embedding_dim)
        x = x.permute(0, 2, 1).contiguous()
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = x.view(batch_size, self.num_posts,
                   self.wordLightBert.word_embedding_dim)  # (batch_size,num_posts,embedding_dim)
        # post transformer
        x = self.postLightBert.post_en(x,
                                       mask=None)  # (batch_size,num_posts,embedding_dim) # (batch_size,num_posts,embedding_dim)
        x = F.adaptive_max_pool1d(x.permute(0, 2, 1).contiguous(), 1).squeeze(-1)  # (batch_size,embedding_dim)
        x = self.post_linear(x)  # (batch_size,out_dim)

        sources = data.source
        sources = self.postLightBert.post_embedding(sources)
        sources = self.postLightBert.post_en(sources, mask=None)
        sources = F.adaptive_max_pool1d(sources.permute(0, 2, 1).contiguous(), 1).squeeze(-1)
        query = copy.deepcopy(sources.detach())
        key = value = copy.deepcopy(x.detach())
        attn = self.Attention(query=query, key=key, value=value, dropout=0.5)

        return th.cat((x, attn), 1)


class InteractionTree(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(InteractionTree, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend, x2), 1)

        x = scatter_mean(x, data.batch, dim=0)

        return x


class RDSPU(th.nn.Module):
    def __init__(self, dataname, in_feats, hid_feats, out_feats, word_embedding_dim, post_embedding_dim, word_head, post_head,
                 word_dff, post_dff, word_droupout, post_droupout, word_N, post_N, num_words, num_posts, out_dim, atten_out_dim):
        super(RDSPU, self).__init__()
        # model = Net(5000, 64, 64).to(device)
        self.StatusesEncoder = StatusesEncoder(dataname=dataname, word_embedding_dim=word_embedding_dim, post_embedding_dim=post_embedding_dim, word_head=word_head, post_head=post_head,
                                               word_dff=word_dff, post_dff=post_dff, word_droupout=word_droupout, post_droupout=post_droupout,
                                               word_N=word_N, post_N=post_N, word_mask=None, post_mask=None, num_words=num_words,
                                               num_posts=num_posts, out_dim=out_dim, atten_out_dim=atten_out_dim)
        self.InteractionTree = InteractionTree(in_feats, hid_feats, out_feats)
        self.fc = th.nn.Linear((out_feats + hid_feats*2) + out_dim + atten_out_dim, 4)

    def forward(self, data):
        SE_x = self.StatusesEncoder(data)
        IT_x = self.InteractionTree(data)
        x = th.cat((SE_x, IT_x), 1)
        x = F.relu(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
