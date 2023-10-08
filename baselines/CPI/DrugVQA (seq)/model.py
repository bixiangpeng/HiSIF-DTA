import torch.nn as nn
import torch
# from torch.autograd import Variable
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5,
                     stride=stride, padding=2, bias=False)
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)
# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv5x5(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.elu(out)
        return out

class DrugVQA(torch.nn.Module):
    """
    The class is an implementation of the DrugVQA model including regularization and without pruning. 
    Slight modifications have been done for speedup


    """
    def __init__(self,args,n_chars_smi,n_chars_seq):
        super(DrugVQA,self).__init__()
        self.batch_size = args.batch_size # 1
        self.lstm_hid_dim = args.lstm_hid_dim #64
        self.r = args.r #10
        self.type = args.task_type #0
        self.in_channels = args.in_channels #8
        #rnn
        self.smile_embeddings = nn.Embedding(n_chars_smi, args.emb_dim) #247,30
        self.seq_embeddings = nn.Embedding(n_chars_seq,args.emb_dim) #21,30
        self.lstm_smile = torch.nn.LSTM(args.emb_dim,self.lstm_hid_dim,2,batch_first=True,bidirectional=True,dropout=args.dropout)
        self.lstm_seq = torch.nn.LSTM(args.emb_dim, self.lstm_hid_dim, 2, batch_first=True, bidirectional=True,dropout=args.dropout)

        self.linear_first_smile = torch.nn.Linear(2*self.lstm_hid_dim,args.d_a)
        self.linear_second_smile = torch.nn.Linear(args.d_a,self.r)

        self.linear_first_seq = torch.nn.Linear(2*self.lstm_hid_dim,args.d_a) #32
        self.linear_second_seq = torch.nn.Linear(args.d_a,self.r)

        #cnn
        # self.conv = conv3x3(1, self.in_channels)
        # self.bn = nn.BatchNorm2d(self.in_channels)
        # self.elu = nn.ELU(inplace=False)
        # self.layer1 = self.make_layer(block, args['cnn_channels'], args['cnn_layers'])
        # self.layer2 = self.make_layer(block, args['cnn_channels'], args['cnn_layers'])

        self.linear_final_step = torch.nn.Linear(self.lstm_hid_dim*2+self.lstm_hid_dim*2,args.dense_hid)
        self.linear_final = torch.nn.Linear(args.dense_hid,args.n_classes)

        # self.smile_hidden_state = self.init_hidden1(64)
        # self.seq_hidden_state = self.init_hidden2(64)
        
    def softmax(self,input, axis=1):
        """
        Softmax applied to axis=n
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied

        Returns:
            softmaxed tensors
        """
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)

    def init_hidden1(self,dim_):
        return (torch.zeros(4,dim_,self.lstm_hid_dim).to('cuda:1'),torch.zeros(4,dim_,self.lstm_hid_dim).to('cuda:1'))
    def init_hidden2(self,dim_):
        return (torch.zeros(4,dim_,self.lstm_hid_dim).to('cuda:1'),torch.zeros(4,dim_,self.lstm_hid_dim).to('cuda:1'))
    
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    # x1 = smiles , x2 = contactMap
    def forward(self,smile_input,seq_input):
        self.smile_hidden_state = self.init_hidden1(smile_input.shape[0])  # 重置隐藏状态
        self.seq_hidden_state = self.init_hidden2(smile_input.shape[0])  # 重置隐藏状态
        #处理字符串
        smile_embed = self.smile_embeddings(smile_input)
        smile_outputs, self.smile_hidden_state = self.lstm_smile(smile_embed,self.smile_hidden_state)
        smile_sentence_att = F.tanh(self.linear_first_smile(smile_outputs))
        smile_sentence_att = self.linear_second_smile(smile_sentence_att)
        smile_sentence_att = self.softmax(smile_sentence_att,1)
        smile_sentence_att = smile_sentence_att.transpose(1,2)
        smile_sentence_embed = smile_sentence_att@smile_outputs
        smile_avg_embed = torch.sum(smile_sentence_embed,1)/self.r  #multi head

        seq_embed = self.seq_embeddings(seq_input)
        seq_outputs, self.seq_hidden_state = self.lstm_seq(seq_embed, self.seq_hidden_state)
        seq_sentence_att = F.tanh(self.linear_first_seq(seq_outputs))
        seq_sentence_att = self.linear_second_seq(seq_sentence_att)
        seq_sentence_att = self.softmax(seq_sentence_att, 1)
        seq_sentence_att = seq_sentence_att.transpose(1, 2)
        seq_sentence_embed = seq_sentence_att @ seq_outputs
        seq_avg_embed = torch.sum(seq_sentence_embed, 1) / self.r  # multi head


         #distance map
        # pic = self.conv(x2)
        # pic = self.bn(pic)
        # pic = self.elu(pic)
        # pic = self.layer1(pic)
        # pic = self.layer2(pic)
        # pic_emb = torch.mean(pic,2)
        # pic_emb = pic_emb.permute(0,2,1)
        # seq_att = F.tanh(self.linear_first_seq(pic_emb))
        # seq_att = self.linear_second_seq(seq_att)
        # seq_att = self.softmax(seq_att,1)
        # seq_att = seq_att.transpose(1,2)
        # seq_embed = seq_att@pic_emb
        # avg_seq_embed = torch.sum(seq_embed,1)/self.r
        
        sscomplex = torch.cat([smile_avg_embed,seq_avg_embed],dim=1)
        sscomplex = F.relu(self.linear_final_step(sscomplex))
        
        if not bool(self.type):
            output = F.sigmoid(self.linear_final(sscomplex))
            return output
        else:
            return F.log_softmax(self.linear_final(sscomplex))