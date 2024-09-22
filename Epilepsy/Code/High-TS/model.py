
import math
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gmp


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    
    def __init__(self, dimension, max_len = 5000):
        super().__init__()

        pe = torch.zeros(max_len, dimension)                             
        position = torch.arange(0, max_len).unsqueeze(1)               
        div_term = torch.exp(torch.arange(0, dimension, 2) *                 
                             -(math.log(100.0) / dimension))                       
        pe[:, 0::2] = torch.sin(position * div_term)                         
        pe[:, 1::2] = torch.cos(position * div_term)                         
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, device, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.device = device
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
    def forward(self, emb_i, emb_j):
        batch_size = emb_i.shape[0]
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)
        
        negatives_mask = torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(self.device).float()

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss_cl = torch.sum(loss_partial) / (2 * batch_size)
        return loss_cl


class HighTS(torch.nn.Module):
    def __init__(self, args, classes_len, win_size_g, node_num_t1, node_num_t2, node_num_t3, device):
        super().__init__()
        
        ### tdl ###
        self.win_size_g = win_size_g
        self.node_num_g = args.node_num_g
        self.gcndimension = args.gcndimension
        self.classes_len = classes_len
        self.device = device
        
        # feature embedding
        self.feature_embedding_g1 = nn.Linear(in_features=win_size_g, out_features=args.gcndimension)
        self.feature_embedding_g2 = nn.Linear(in_features=1, out_features=args.gcndimension)
        self.feature_embedding_g3 = nn.Linear(in_features=win_size_g+4, out_features=args.gcndimension)
        # position embedding
        self.positional_encoding_g = PositionalEncoding(args.gcndimension)    
        
        # gcn
        self.gcn1 = GCNConv(args.gcndimension, args.gcndimension)
        self.gcn2 = GCNConv(args.gcndimension, args.gcndimension)
        self.gcn3 = GCNConv(args.gcndimension, args.gcndimension)
        
        # activation and regularization
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        ### multiscale transformer ###
        self.T_FE_DIM = args.T_FE_DIM
        self.node_num_t1 = node_num_t1
        self.node_num_t2 = node_num_t2
        self.node_num_t3 = node_num_t3
        
        # feature embedding
        self.feature_embedding_t1 = nn.Linear(in_features=args.win_size_t1, out_features=args.T_FE_DIM)
        self.feature_embedding_t2 = nn.Linear(in_features=args.win_size_t2, out_features=args.T_FE_DIM)
        self.feature_embedding_t3 = nn.Linear(in_features=args.win_size_t3, out_features=args.T_FE_DIM)
        # position embedding
        self.positional_encoding_t = PositionalEncoding(args.T_FE_DIM)
        
        # transformer encoder
        encoder_layer1 = nn.TransformerEncoderLayer(args.T_FE_DIM, args.T_NUM_HEADS, batch_first=True)
        encoder_layer2 = nn.TransformerEncoderLayer(args.T_FE_DIM, args.T_NUM_HEADS, batch_first=True)
        encoder_layer3 = nn.TransformerEncoderLayer(args.T_FE_DIM, args.T_NUM_HEADS, batch_first=True)
        self.transformer_encoder1 = nn.TransformerEncoder(encoder_layer1, args.T_NUM_ENCODER)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer2, args.T_NUM_ENCODER)
        self.transformer_encoder3 = nn.TransformerEncoder(encoder_layer3, args.T_NUM_ENCODER)
        
        # contrastive learning
        self.ContrastiveLoss = ContrastiveLoss(device)
        
        # output
        self.output_layer = nn.Linear(args.gcndimension*args.num_order+args.T_FE_DIM*args.T_NUM_NET, classes_len)


    def forward(self, data_g1, data_g2, data_g3, data_t1, data_t2, data_t3):
        
        ### tdl ##
        # print("# ================= 0 simplex ===================== #")
        seg = data_g1.x.to(self.device)
        fe_g1 = self.feature_embedding_g1(seg)
        fe_g1 = torch.reshape(fe_g1,[-1, self.node_num_g, self.gcndimension])
        representation_1 = self.positional_encoding_g(fe_g1)  
        repre_1 = torch.reshape(representation_1, [-1, self.gcndimension])
        edge_index_g1 = data_g1.edge_index.to(self.device)
        
        x_g1 = self.gcn1(repre_1,edge_index_g1)
        x_g1 = self.relu(x_g1)

        batch_g1 = data_g1.batch.to(self.device)
        x_g1 = gmp(x_g1, batch_g1)
        
        # print("# ================= 1 simplex ===================== #")
        cosines = data_g2.x.to(self.device)
        fe_g2 = self.feature_embedding_g2(cosines)
        edge_index_g2 = data_g2.edge_index.to(self.device)
        
        x_g2 = self.gcn2(fe_g2,edge_index_g2)
        x_g2 = self.relu(x_g2)

        batch_g2 = data_g2.batch.to(self.device)        
        x_g2 = gmp(x_g2, batch_g2)

        # print("# ================= 2 simplex ===================== #")
        triangles = data_g3.x.to(self.device)
        fe_g3 = self.feature_embedding_g3(triangles)
        edge_index_g3 = data_g3.edge_index.to(self.device)
        
        x_g3 = self.gcn3(fe_g3,edge_index_g3)
        x_g3 = self.relu(x_g3)

        batch_g3 = data_g3.batch.to(self.device)        
        x_g3 = gmp(x_g3, batch_g3)
        
        ### multiscale transformer ###
        # print("# ================= 1 scale ===================== #")
        fe_t1 = self.feature_embedding_t1(data_t1.x.to(self.device))
        fe_t1 = torch.reshape(fe_t1, [-1,self.node_num_t1,self.T_FE_DIM])
        src_t1 = self.positional_encoding_t(fe_t1)
        
        x_t1 = self.transformer_encoder1(src_t1) 
        x_t1 = torch.reshape(x_t1, [-1, self.T_FE_DIM])
        
        batch_t1 = data_t1.batch.to(self.device)
        x_t1 = gmp(x_t1, batch_t1)
        
        # print("# ================= 2 scale ===================== #")
        fe_t2 = self.feature_embedding_t2(data_t2.x.to(self.device))
        fe_t2 = torch.reshape(fe_t2, [-1,self.node_num_t2,self.T_FE_DIM])
        src_t2 = self.positional_encoding_t(fe_t2)
    
        x_t2 = self.transformer_encoder2(src_t2) 
        x_t2 = torch.reshape(x_t2, [-1, self.T_FE_DIM])

        batch_t2 = data_t2.batch.to(self.device)
        x_t2 = gmp(x_t2, batch_t2)
        
        # print("# ================= 3 scale ===================== #")
        fe_t3 = self.feature_embedding_t3(data_t3.x.to(self.device))
        fe_t3 = torch.reshape(fe_t3, [-1,self.node_num_t3,self.T_FE_DIM])
        src_t3 = self.positional_encoding_t(fe_t3)
        
        x_t3 = self.transformer_encoder3(src_t3) 
        x_t3 = torch.reshape(x_t3, [-1, self.T_FE_DIM])

        batch_t3 = data_t3.batch.to(self.device)
        x_t3 = gmp(x_t3, batch_t3)
        
        # print("# =========== contrastive learning ============== #")
        emb_i = torch.cat((x_g1, x_g2, x_g3), 1)
        emb_j = torch.cat((x_t1, x_t2, x_t3), 1)
        loss_cl = self.ContrastiveLoss(emb_i, emb_j)
        
        # print("# ================= cat ===================== #")
        x = torch.cat((x_g1, x_g2, x_g3, x_t1, x_t2, x_t3), 1)

        # print("# ================= out ===================== #")
        pred = self.output_layer(x)
        
        return pred, loss_cl