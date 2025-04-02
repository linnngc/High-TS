
import torch
import numpy as np

# preprocessing

def encode_onehot(labels):                                                   
    classes = set(labels)                                                    
    classes_len = len(classes)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot, classes_len

def load_data(path, dataset):    
    labels_series = np.genfromtxt("{}{}.txt".format(path, dataset),         
                                        dtype=np.dtype(str))
    labels_, classes_len = encode_onehot(labels_series[:, 0])
    labels_ = torch.Tensor(labels_)
    series = labels_series[:, 1:]
    series = np.array(series, dtype=np.float32)
    series = torch.Tensor(series)                
    return series, labels_, classes_len

def segment(series, sample_num, win_size, node_num):
    seg = torch.zeros((sample_num, win_size, node_num))                     
    for i in range(node_num):
        seg[:,:,i] = series[:,i*win_size:(i+1)*win_size]                      
    seg_= seg.transpose(1,2).to(torch.float32)
    return seg_
        

# 0-simplex
def zero_odr_edge(seg_, k, sample_num, node_num):
    zero_odr_edges = []
    zero_odr_cosine = []
    for i in range(sample_num):
        s = seg_[i,:,:] / torch.norm(seg_[i,:,:], dim=-1, keepdim=True)
        cos_sim = torch.mm(s, s.T) 
        cos_sim[np.tril_indices(node_num)] = float("-inf")                   
        cos_sim[torch.round(cos_sim, decimals=4)==1] = float("-inf")
        cos_sim = cos_sim.reshape(-1)                                        
        values, indices = torch.topk(cos_sim, k, largest=True)               
        a_indices = torch.zeros(node_num*node_num)
        a_indices = a_indices.scatter(0,indices, 1)                         
        adj_zero = torch.reshape(a_indices, [node_num,node_num])            
        adj_zero = adj_zero.numpy()
        zero_odr_edges.append(torch.tensor(np.argwhere(adj_zero==1)).T.int())
        indices_sort = torch.argsort(indices)                                
        values_sort = torch.index_select(values, 0, indices_sort)            
        zero_odr_cosine.append(values_sort.unsqueeze(1))                    
    return zero_odr_edges, zero_odr_cosine

# 1-simplex
def one_odr_edge(zero_odr_edges, k, sample_num):
    one_odr_edges = []
    for i in range(sample_num):
        adj_one = torch.zeros(k,k)
        for j in range(k-1):                                                 
            tensor_isin_one = torch.zeros(k,2)
            for jj in range(j+1,k):
                tensor_isin_one[jj,:] = torch.isin(zero_odr_edges[i].T[j],zero_odr_edges[i].T[jj])
                if tensor_isin_one[jj].sum()==1:                             
                    adj_one[j,jj]=1                                         
        adj_one = adj_one.numpy()
        one_odr_edges.append(torch.Tensor(np.argwhere(adj_one==1)).T.int())  
    return one_odr_edges

# 2-simplex 
def two_odr_triangle(zero_odr_edges, one_odr_edges, sample_num, k):
    triangle = []
    for i in range(sample_num):
        mat_1 = zero_odr_edges[i][:,one_odr_edges[i][0].long()].T                   
        mat_2 = zero_odr_edges[i][:,one_odr_edges[i][1].long()].T                  
        mat_3 = zero_odr_edges[i].T                                        
        tensor_isin_tri = []
        for j in range(len(one_odr_edges[i][0])):
            for jj in range(k):
                if  torch.isin(mat_3[jj], mat_1[j]).sum() == 1 and torch.isin(mat_3[jj], mat_2[j]).sum() == 1:
                    nodes_set = set(torch.cat((mat_1[j],mat_2[j],mat_3[jj])).numpy())
                    if len(nodes_set) == 3:
                        tensor_isin_tri.append(nodes_set)
                    else:
                        pass
                else:
                    pass
        three_nodes = []
        [three_nodes.append(item) for item in tensor_isin_tri if not item in three_nodes]
        temp = torch.zeros(len(three_nodes), 3)
        for n in range(len(three_nodes)):
            temp[n] = torch.tensor(list(three_nodes[n]))
        triangle.append(temp.int())
    return triangle

def two_odr_edge(triangle, sample_num):
    two_odr_edges = []
    for i in range(sample_num):
        adj_two = torch.zeros(len(triangle[i]),len(triangle[i]))
        for j in range(len(triangle[i])-1):                             
            tensor_isin_two = torch.zeros(len(triangle[i]),3)
            for jj in range(j+1,len(triangle[i])):                      
                tensor_isin_two[jj,:] = torch.isin(triangle[i][j],triangle[i][jj])
                if tensor_isin_two[jj].sum()==2:                         
                    adj_two[j,jj]=1
        adj_two = adj_two.numpy()
        two_odr_edges.append(torch.Tensor(np.argwhere(adj_two==1)).T.int())   
    return two_odr_edges

def triangle_feature(seg_, triangle, sample_num, win_size):
    triangle_features = []
    for i in range(sample_num):
        triangle_seg = seg_[i,triangle[i].long()]
        triangle_center = torch.sum(triangle_seg, 1)/3
        triangle_cos = torch.zeros(len(triangle_seg),3)
        triangle_s = torch.zeros(len(triangle_seg),1)
        for j in range(len(triangle_seg)):
            triangle_edge = torch.zeros(3,win_size)
            triangle_edge[0] = triangle_seg[j][0]-triangle_seg[j][1]
            triangle_edge[1] = triangle_seg[j][1]-triangle_seg[j][2]
            triangle_edge[2] = triangle_seg[j][2]-triangle_seg[j][0]
            norm_2 = torch.norm(triangle_edge, dim=-1, keepdim=True)
            s = triangle_edge / norm_2
            cos_ = torch.mm(s, s.T) 
            triangle_cos[j] = torch.tensor([cos_[0,1], cos_[0,2], cos_[1,2]])
            p = norm_2.sum()/2
            triangle_s[j] = np.sqrt(p*(p-norm_2[0])*(p-norm_2[1])*(p-norm_2[2]))
        triangle_features.append(torch.cat((triangle_center, triangle_cos, triangle_s),1))
    return triangle_features