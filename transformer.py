# 导入相关模块
import torch
import math
from torch import nn
import ml_collections
import copy

#2.构建self-Attention模块
class Attention(nn.Module):
    def __init__(self,config,vis):
        super(Attention,self).__init__()
        self.vis=vis
        self.num_attention_heads=config.transformer["num_heads"]#12
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)  # 768/12=64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 12*64=768

        self.query = nn.Linear(config.hidden_size, self.all_head_size)#wm,768->768，Wq矩阵为（768,768）
        self.key = nn.Linear(config.hidden_size, self.all_head_size)#wm,768->768,Wk矩阵为（768,768）
        self.value = nn.Linear(config.hidden_size, self.all_head_size)#wm,768->768,Wv矩阵为（768,768）
        self.out = nn.Linear(config.hidden_size, config.hidden_size)  # wm,768->768
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
        self.num_attention_heads, self.attention_head_size)  # wm,(bs,197)+(12,64)=(bs,197,12,64)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # wm,(bs,12,197,64)

    def forward(self, hidden_states):
        # hidden_states为：(bs,197,768)
        mixed_query_layer = self.query(hidden_states)#wm,768->768
        mixed_key_layer = self.key(hidden_states)#wm,768->768
        mixed_value_layer = self.value(hidden_states)#wm,768->768

        query_layer = self.transpose_for_scores(mixed_query_layer)#wm，(bs,12,197,64)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))#将q向量和k向量进行相乘（bs,12,197,197)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)#将结果除以向量维数的开方
        attention_probs = self.softmax(attention_scores)#将得到的分数进行softmax,得到概率
        weights = attention_probs if self.vis else None#wm,实际上就是权重
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)#将概率与内容向量相乘
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)#wm,(bs,197)+(768,)=(bs,197,768)
        context_layer = context_layer.view(*new_context_layer_shape)



        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights#wm,(bs,197,768),(bs,197,197)


#3.构建前向传播神经网络
#两个全连接神经网络，中间加了激活函数
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer["mlp_dim"])#wm,786->3072
        self.fc2 = nn.Linear(config.transformer["mlp_dim"], config.hidden_size)#wm,3072->786
        self.act_fn = torch.nn.functional.gelu#wm,激活函数
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)#wm,786->3072
        x = self.act_fn(x)#激活函数
        x = self.dropout(x)#wm,丢弃
        x = self.fc2(x)#wm3072->786
        x = self.dropout(x)
        return x


# 4.构建编码器的可重复利用的Block()模块：每一个block包含了self-attention模块和MLP模块
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size  # wm,768
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)  # wm，层归一化
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h  # 残差结构

        hh = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + hh  # 残差结构
        return x, weights

#5.构建Encoder模块，该模块实际上就是堆叠N个Block模块
class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights
#6构建transformers完整结构，首先图片被embedding模块编码成序列数据，然后送入Encoder中进行编码
class Transformer(nn.Module):
    def __init__(self,config, vis):
        super(Transformer, self).__init__()
        # self.embeddings = Embeddings(config)#wm,对一幅图片进行切块编码，得到的是（bs,n_patch+1（196）,每一块的维度（768））
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        # embedding_output = self.embeddings(input_ids)#wm,输出的是（bs,196,768)
        # encoded, attn_weights = self.encoder(embedding_output)#wm,输入的是（bs,196,768)
        encoded, attn_weights = self.encoder(input_ids)
        return encoded, attn_weights#输出的是（bs,197,768

class tmain(nn.Module):
    def __init__(self,hidden_size,num_layers,num_heads):
        super(tmain, self).__init__()
        config = ml_collections.ConfigDict()
        # config.patches = ml_collections.ConfigDict({'size':4})
        config.hidden_size = hidden_size
        config.transformer = ml_collections.ConfigDict()
        config.transformer.mlp_dim = 2048
        config.transformer.num_heads = num_heads
        config.transformer.num_layers = num_layers
        config.transformer.attention_dropout_rate = 0.0
        config.transformer.dropout_rate = 0.2
        config.classifier = 'token'
        config.representation_size = None
        self.transformers = Transformer(config,vis=True)
        # self.conv_1 = nn.Conv2d(3, 64, kernel_size=1, padding=0)

        # self.transformers1 = self.make_layers(64, 2, 2, ['M', 128])
        # self.transformers2 = self.make_layers(64, 4, 4,  ['M', 256])
        # self.transformers3 = self.make_layers(64, 8, 8,  ['M', 512, 512])
        # self.transformers4 = self.make_layers(64, 16, 16,  ['M', 512, 512])
        # self.transformers3 = self.make_layers(256, 2, 2, ['M', 512])
        # self.transformers4 = self.make_layers(512, 2, 2,  ['M', 512])
        # self.transformers = Transformer(con_fig(), vis=True)
    def forward(self, x):
        # a, b, c, d = x.shape
        # a, b, c, d = x[1].shape
        # img = x[1].view(a,  c*d, b)
        # img = x.view(a, b, c * d)
        out_transformers, _ = self.transformers(x)
        # out_transformers = out_transformers.view(a, b, c, d)
        # conv_1 = self.conv_1(out_transformers)#64
        # out_transformers_pool1 = self.transformers1(out_transformers)#128
        # out_transformers_pool2 = self.transformers2(out_transformers)#256
        # out_transformers_pool3 = self.transformers3(conv_1)#512
        # out_transformers_pool4 = self.transformers4(conv_1)#512
        # out_transformers_pool3 = self.transformers3(out_transformers)  # 512
        # out_transformers_pool4 = self.transformers4(out_transformers_pool3)  # 512
        # return x[0], out_transformers, out_transformers_pool3, out_transformers_pool4
        # return x,conv_1, out_transformers_pool1, out_transformers_pool2
        # return x, out_transformers_pool2,out_transformers_pool3, out_transformers_pool4
        # return x, conv_1, out_transformers_pool1, out_transformers_pool2, out_transformers_pool3, out_transformers_pool4
        return out_transformers
    @staticmethod
    #定义的实现卷积层的模块化函数
    def make_layers(in_channels, kernel_size1, stride1, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=kernel_size1, stride=stride1)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=1, padding=0, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
