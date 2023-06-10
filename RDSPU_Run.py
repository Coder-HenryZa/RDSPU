import sys, os

sys.path.append(os.getcwd())
from util.process import *
import torch as th
import torch.nn.functional as F
import numpy as np
from util.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from util.loadSplitData import *

import matplotlib.pyplot as plt
from datetime import datetime
from networks.RDSPU import RDSPU
import argparse
import warnings

warnings.filterwarnings("ignore")


def run_model(treeDic, x_test, x_train, droprate, lr, weight_decay, patience, n_epochs, batchsize,
              dataname, word_embedding_dim, post_embedding_dim, word_head, post_head, word_dff, post_dff, word_droupout,
              post_droupout, word_N, post_N, num_words, num_posts, out_dim, in_feats, hid_feats, out_feats,
              atten_out_dim):
    model = RDSPU(dataname, in_feats, hid_feats, out_feats, word_embedding_dim, post_embedding_dim, word_head, post_head,
                  word_dff, post_dff,
                  word_droupout, post_droupout, word_N, post_N, num_words, num_posts, out_dim, atten_out_dim).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    early_stopping = EarlyStopping(patience=patience, verbose=True)
    traindata_list, testdata_list = loadUdData(dataname, treeDic, x_train, x_test, droprate)
    for epoch in range(n_epochs):

        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=1)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=1)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            out_labels = model(Batch_data)
            finalloss = F.nll_loss(out_labels, Batch_data.y)
            loss = finalloss
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            print("Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc))
            batch_idx = batch_idx + 1

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []

        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            val_out = model(Batch_data)
            val_loss = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            temp_val_accs.append(val_acc)

        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))

        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))

        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), model, 'RDMSC', dataname)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    show_val = list(val_accs)

    dt = datetime.now()
    save_time = dt.strftime('%Y_%m_%d_%H_%M_%S')

    fig = plt.figure()
    plt.plot(range(1, len(train_accs) + 1), train_accs, color='b', label='train')
    plt.plot(range(1, len(show_val) + 1), show_val, color='r', label='dev')
    plt.grid()
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.xticks(np.arange(1, len(train_accs), step=4))
    fig.savefig('result/' + '{}_accuracy_{}.png'.format(dataname, save_time))

    fig = plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, color='b', label='train')
    plt.plot(range(1, len(val_losses) + 1), val_losses, color='r', label='dev')
    plt.grid()
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.xticks(np.arange(1, len(train_losses) + 1, step=4))
    fig.savefig('result/' + '{}_loss_{}.png'.format(dataname, save_time))

    return train_losses, val_losses, train_accs, val_accs


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    print("seed:", seed)


parser = argparse.ArgumentParser(description='RDMSC')
parser.add_argument('--lr', default=0.0005, type=float, help='Learning Rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay coefficient')
parser.add_argument('--patience', default=10, type=int, help='Early Stopping')
parser.add_argument('--n_epochs', default=200, type=int, help='Training Epochs')
parser.add_argument('--batchsize', default=128, type=int, help='Batch Size')
parser.add_argument('--droprate', default=0.2, type=float, help='Randomly invalidate some edges')
parser.add_argument('--seed', default=12, type=int)

parser.add_argument('--in_feats', default=5000, type=int)
parser.add_argument('--hid_feats', default=64, type=int)
parser.add_argument('--out_feats', default=64, type=int)

parser.add_argument('--word_embedding_dim', default=128, type=int)
parser.add_argument('--post_embedding_dim', default=128, type=int)
parser.add_argument('--word_head', default=2, type=int)
parser.add_argument('--post_head', default=2, type=int)
parser.add_argument('--word_dff', default=128, type=int)
parser.add_argument('--post_dff', default=128, type=int)
parser.add_argument('--word_droupout', default=0.2, type=float)
parser.add_argument('--post_droupout', default=0.2, type=float)
parser.add_argument('--word_N', default=1, type=int)
parser.add_argument('--post_N', default=1, type=int)
parser.add_argument('--num_words', default=35, type=int)
parser.add_argument('--num_posts', default=20, type=int)
parser.add_argument('--out_dim', default=128, type=int)
parser.add_argument('--atten_out_dim', default=4, type=int)

args = parser.parse_args()

if __name__ == '__main__':

    set_seed(args.seed)
    datasetname = "Twitter15"  # Twitter15 Twitter16
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    test_set, train_set = loadSplitData(datasetname)
    treeDic = loadTree(datasetname)
    train_losses, val_losses, train_accs, val_accs = run_model(treeDic, test_set, train_set, args.droprate, args.lr,
                                                               args.weight_decay, args.patience, args.n_epochs,
                                                               args.batchsize, datasetname, args.word_embedding_dim,
                                                               args.post_embedding_dim, args.word_head,
                                                               args.post_head, args.word_dff, args.post_dff,
                                                               args.word_droupout, args.post_droupout, args.word_N,
                                                               args.post_N, args.num_words, args.num_posts,
                                                               args.out_dim, args.in_feats, args.hid_feats,
                                                               args.out_feats, args.atten_out_dim)
    print("Total_Best_Accuracy:{}".format(max(val_accs)))
