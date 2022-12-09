import torch.nn as nn
import torch
import torch.nn.functional as F
from ..model import Model

class LSTM(Model):
    def __init__(self, config):
        num_classes = config['num_classes']
        hidden_size = config['hidden_size']
        embedding_size = config['embedding_size']
        vocab_size = self.get_vocab_size(config)

        super(LSTM,self).__init__(config)
        self.return_embedding = config['return_embedding']
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, dropout=0.5)
        temp = nn.Linear(hidden_size, num_classes, bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)
    
    def get_vocab_size(self, config):
        return 30522

    def forward(self,input):
        '''
        :param input: 
        :return: 
        '''
        input_embeded = self.embedding(input)    #[batch_size,seq_len,200]

        output,h_n = self.lstm(input_embeded)
        output = output[:, -1, :]

        #全连接
        if self.return_embedding:
            return output
        else:
            out = torch.matmul(output, self.prototype.T)
            return F.log_softmax(out)