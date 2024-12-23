import torch
from torch import nn
import torchvision
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageEncoder(nn.Module):

    def __init__(self, encoded_image_size=14, d_model=512):
        super(ImageEncoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.d_model = d_model

        # Load ResNet-101 and remove the final layers
        resnet = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Adaptive pooling to resize feature maps
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # Linear layer to project ResNet features to d_model
        self.linear_projection = nn.Linear(2048, d_model)
        self.init_weights()

        # Fine-tuning control
        self.fine_tune()

    def forward(self, images):

        out = self.resnet(images)  # Shape: (batch_size, 2048, H, W)
        out = self.adaptive_pool(out)  # Shape: (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # Shape: (batch_size, H, W, 2048)

        # Flatten spatial dimensions and project features
        out = out.view(out.size(0), -1, 2048)  # Shape: (batch_size, seq_len, 2048)
        out = self.linear_projection(out)  # Shape: (batch_size, seq_len, d_model)  seq_len= 14*14=196

        return out

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
    
    def init_weights(self):
        for name, param in self.linear_projection.named_parameters():
            if 'weight' in name:  
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.01)





class InputEmbeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size # unique no of token 
        self.embedding=nn.Embedding(vocab_size,d_model) # embeding table ---> (vocab_size,d_model)
    
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int ,seq_len:int,dropout:float):
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        pe=torch.zeros(seq_len,d_model)
        print(pe[:2,:3].shape)
        position=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1) #  seq_len---->(seq_len,1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        pe=pe.unsqueeze(0) #(1,seq_len,d_model)
        self.register_buffer('pe',pe)
        # print(position.shape)

    def forward(self,x):
        x=x+(self.pe[:,:x.shape[1],:]).requires_grad_(False) # only till x_len  with broadcasting
        return  self.dropout(x) #only during training

class LayerNormalization(nn.Module):
    def __init__(self,eps:float=10**-6):
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1))
        self.bias=nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True) # batch,x.shape[1],1   , broadcasting here also 
        std=x.std(dim=-1,keepdim=True)
        return self.alpha*(x-mean)/(std+self.eps) + self.bias

    
class MLP(nn.Module):
    def __init__(self,d_model:int, d_ff:int, dropout:int):
        super().__init__()
        self.linear_1=nn.Linear(d_model,d_ff)
        self.dropout=nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff,d_model)
    
    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model:int, h:int,dropout:float):
        super().__init__()
        self.d_model=d_model
        self.h=h
        assert d_model%h==0,"d_model is not divisible by h"

        self.d_k=d_model//h
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)

        self.w_o=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)

    @staticmethod
    
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k=query.shape[-1]

        attention_scores=(query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask==0 ,-1e9)
        attention_scores=attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores=dropout(attention_scores)

        return (attention_scores @ value), attention_scores


    def forward(self,q,k,v,mask):
        query=self.w_q(q)
        key=self.w_k(k)
        value=self.w_v(v)

        query=query.view(query.shape[0],query.shape[1],self.h ,self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.h ,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.h ,self.d_k).transpose(1,2)

        x,self.attention_scores=MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)

        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self,dropout:float):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization()
    
    def forward(self,x,sublayer):
        return x+ self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttentionBlock,feed_forward_block:MLP,dropout:float):
        super().__init__()
        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connection=nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self,x):
        x=self.residual_connection[0](x,lambda x:self.self_attention_block(x,x,x,mask=None)) # output= INPUT+Dropout(Sublayer(Norm(x)))
        x=self.residual_connection[1](x,self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()
    
    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttentionBlock,cross_attention_block:MultiHeadAttentionBlock,feed_forward_block:MLP,dropout:float):
        super().__init__()
        self.self_attention_block=self_attention_block
        self.cross_attention_block=cross_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connection=nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    
    def forward(self,x,encoder_output,tgt_mask):
        x=self.residual_connection[0](x,lambda x:self.self_attention_block(x,x,x,tgt_mask)) # output= INPUT+Dropout(Sublayer(Norm(x)))
        x=self.residual_connection[1](x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output,mask=None))
        x=self.residual_connection[2](x,self.feed_forward_block)
        return x
class Decoder(nn.Module):
    def __init__(self,layers:nn.Module):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()

    def forward(self,x,encoder_output,tgt_mask):
        for layer in self.layers:
            x=layer(x,encoder_output,tgt_mask)
        return self.norm(x)
     
class ProjectionLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.proj=nn.Linear(d_model,vocab_size)
    
    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1)
    
class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,tgt_embed:InputEmbeddings,tgt_pos:PositionalEncoding,projection_layer:ProjectionLayer):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.target_embed=tgt_embed
        self.tgt_pos=tgt_pos
        self.projection_layer=projection_layer

    def encode(self,src):
        return self.encoder(src)
    
    def decode(self,encoder_output,tgt,tgt_mask):
        tgt=self.target_embed(tgt)
        tgt=self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,tgt_mask)

    def projection (self,x):
        return self.projection_layer(x)
    def beta(self):
       return 4


def build_transformer(tgt_vocab_size:int,tgt_seq_len:int,d_model:int=512,N:int=6,h:int=8,dropout:float=0.1,d_ff:int=2048):
    tgt_embed=InputEmbeddings(d_model,tgt_vocab_size)

    tgt_pos=PositionalEncoding(d_model,tgt_seq_len,dropout)

    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention_block=MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block=MLP(d_model,d_ff,dropout)
        encoder_block=EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)
    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention_block=MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block=MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block=MLP(d_model,d_ff,dropout)
        decoder_block=DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)

    encoder=Encoder(nn.ModuleList(encoder_blocks))
    decoder=Decoder(nn.ModuleList(decoder_blocks))

    projection_layer=ProjectionLayer(d_model,tgt_vocab_size)

    transformer=Transformer(encoder,decoder,tgt_embed,tgt_pos,projection_layer)

    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform(p)

    return transformer










# class DecoderWithAttention(nn.Module):
#     """
#     Decoder.
#     """

#     def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
#         """
#         :param attention_dim: size of attention network
#         :param embed_dim: embedding size
#         :param decoder_dim: size of decoder's RNN
#         :param vocab_size: size of vocabulary
#         :param encoder_dim: feature size of encoded images
#         :param dropout: dropout
#         """
#         super(DecoderWithAttention, self).__init__()

#         self.encoder_dim = encoder_dim
#         self.attention_dim = attention_dim
#         self.embed_dim = embed_dim
#         self.decoder_dim = decoder_dim
#         self.vocab_size = vocab_size
#         self.dropout = dropout

#         self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

#         self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
#         self.dropout = nn.Dropout(p=self.dropout)
#         self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
#         self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
#         self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
#         self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
#         self.sigmoid = nn.Sigmoid()
#         self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
#         self.init_weights()  # initialize some layers with the uniform distribution

#     def init_weights(self):
#         self.embedding.weight.data.uniform_(-0.1, 0.1)
#         self.fc.bias.data.fill_(0)
#         self.fc.weight.data.uniform_(-0.1, 0.1)

#     def load_pretrained_embeddings(self, embeddings):

#         self.embedding.weight = nn.Parameter(embeddings)

#     def fine_tune_embeddings(self, fine_tune=True):

#         for p in self.embedding.parameters():
#             p.requires_grad = fine_tune

#     def init_hidden_state(self, encoder_out):
#         mean_encoder_out = encoder_out.mean(dim=1)
#         h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
#         c = self.init_c(mean_encoder_out)
#         return h, c

#     def forward(self, encoder_out, encoded_captions, caption_lengths):
#         """
#         Forward propagation.

#         :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
#         :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
#         :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
#         :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
#         """

#         batch_size = encoder_out.size(0)
#         encoder_dim = encoder_out.size(-1)
#         vocab_size = self.vocab_size

#         # Flatten image
#         encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
#         num_pixels = encoder_out.size(1)

#         # Sort input data by decreasing lengths; why? apparent below
#         caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
#         encoder_out = encoder_out[sort_ind]
#         encoded_captions = encoded_captions[sort_ind]

#         # Embedding
#         embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

#         # Initialize LSTM state
#         h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

#         # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
#         # So, decoding lengths are actual lengths - 1
#         decode_lengths = (caption_lengths - 1).tolist()

#         # Create tensors to hold word predicion scores and alphas
#         predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
#         alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

#         # At each time-step, decode by
#         # attention-weighing the encoder's output based on the decoder's previous hidden state output
#         # then generate a new word in the decoder with the previous word and the attention weighted encoding
#         for t in range(max(decode_lengths)):
#             batch_size_t = sum([l > t for l in decode_lengths])
#             attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
#                                                                 h[:batch_size_t])
#             gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
#             attention_weighted_encoding = gate * attention_weighted_encoding
#             h, c = self.decode_step(
#                 torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
#                 (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
#             preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
#             predictions[:batch_size_t, t, :] = preds
#             alphas[:batch_size_t, t, :] = alpha

#         return predictions, encoded_captions, decode_lengths, alphas, sort_ind
