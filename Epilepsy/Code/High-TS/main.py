
import torch
import torch.nn as nn
import numpy as np
import argparse
from sklearn.metrics import accuracy_score
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from dataprocess import load_data, segment, zero_odr_edge, one_odr_edge, two_odr_triangle, two_odr_edge, triangle_feature
from model import HighTS
import warnings
warnings.filterwarnings('ignore') 


def DATA_g(series, node_num, percentage):    
    sample_num = series.shape[0]
    win_size = int(len(series[0])/node_num)                           
    seg_ = segment(series, sample_num, win_size, node_num)
    k = int(node_num*node_num*percentage)                                    
    zero_odr_edges, cosines = zero_odr_edge(seg_, k, sample_num, node_num)
    one_odr_edges = one_odr_edge(zero_odr_edges, k, sample_num)
    two_odr_triangles = two_odr_triangle(zero_odr_edges, one_odr_edges, sample_num, k)
    triangle_features = triangle_feature(seg_, two_odr_triangles, sample_num, win_size)
    two_odr_edges = two_odr_edge(two_odr_triangles, sample_num)
    return seg_, zero_odr_edges, cosines, one_odr_edges, triangle_features, two_odr_edges, classes_len, win_size, sample_num

def DATA_t(series, sample_num, win_size_1, win_size_2, win_size_3):
    node_num_1 = int(len(series[0])/win_size_1)                        
    seg_1_ = segment(series, sample_num, win_size_1, node_num_1)
    node_num_2 = int(len(series[0])/win_size_2)                        
    seg_2_ = segment(series, sample_num, win_size_2, node_num_2)
    node_num_3 = int(len(series[0])/win_size_3)                        
    seg_3_ = segment(series, sample_num, win_size_3, node_num_3)
    return seg_1_, node_num_1, seg_2_, node_num_2, seg_3_, node_num_3
  
def batch_g(src, labels_, edges, sample_num, batch_size):
    dataset = []
    for i in range(sample_num):
        data = Data(x = src[i], edge_index = edges[i], y = labels_[i].unsqueeze(0))
        dataset.append(data)
    loader = DataLoader(dataset, batch_size, shuffle=False)
    return loader

def batch_t(seg_, sample_num, batch_size):
    dataset = []
    for i in range(sample_num):
        data = Data(x = seg_[i])
        dataset.append(data)
    loader = DataLoader(dataset, batch_size, shuffle=False)
    return loader

def train(model, device, loader_train_g1, loader_train_g2, loader_train_g3, 
          loader_train_t1, loader_train_t2, loader_train_t3, optimizer, classes_len):
    model.train()    
    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    loss_accum = 0
    for batch_idx, data in enumerate(zip(loader_train_g1, loader_train_g2, loader_train_g3, 
                                         loader_train_t1, loader_train_t2, loader_train_t3)):
        data_g1, data_g2, data_g3, data_t1, data_t2, data_t3 = \
            data[0], data[1], data[2], data[3], data[4], data[5]
        optimizer.zero_grad()
        y = data_g1.y  
        y_ = torch.Tensor(np.where(y)[1])
        y_true_ = y_.long().to(device)
        out, loss_cl = model(data_g1, data_g2, data_g3, data_t1, data_t2, data_t3)
        loss = criterion(out, y_true_)+loss_cl
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().cpu().item() 
        preds = torch.cat([preds, out.detach()], dim=0)
        targets = torch.cat([targets, y_true_], dim=0) 
    return loss_accum, preds, targets
 
def predict(model, device, loader_test_g1, loader_test_g2, loader_test_g3, 
            loader_test_t1, loader_test_t2, loader_test_t3, classes_len):
    model.eval()
    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, data in enumerate(zip(loader_test_g1, loader_test_g2, loader_test_g3, 
                                             loader_test_t1, loader_test_t2, loader_test_t3)):
            data_g1, data_g2, data_g3, data_t1, data_t2, data_t3 = \
                data[0], data[1], data[2], data[3], data[4], data[5]
            y = data_g1.y
            y = torch.Tensor(np.where(y)[1])
            y_true = y.to(device)
            out, loss_cl = model(data_g1, data_g2, data_g3, data_t1, data_t2, data_t3)
            preds = torch.cat([preds, out.detach()], dim=0)
            targets = torch.cat([targets, y_true], dim=0) 
    return preds, targets


if __name__ == "__main__":
                                 
    print("main")
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--file_name', type=str, default='C&E', help='dataset name')     
    parser.add_argument('--NUM_EPOCHS', type=int, default=3001, help = 'number of epochs')
    parser.add_argument('--LR', type=float, default=0.0002, help = 'learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help = 'batch size')
    parser.add_argument('--ie', type=int, default=5, help = 'number of independent experiments')
    # multiscale transformer
    parser.add_argument('--T_FE_DIM', type=int, default=64, help = 'dimension of latent representation')  
    parser.add_argument('--win_size_t1', type=int, default=1, help = 'window size of 1-scale')
    parser.add_argument('--win_size_t2', type=int, default=2, help = 'window size of 2-scale')
    parser.add_argument('--win_size_t3', type=int, default=3, help = 'window size of 3-scale')
    parser.add_argument('--T_NUM_NET', type=int, default=3, help = 'number of scales')
    parser.add_argument('--T_NUM_HEADS', type=int, default=2, help = 'number of transformer heads')
    parser.add_argument('--T_NUM_ENCODER', type=int, default=2, help = 'number of transformer encoder layers')
    # tdl
    parser.add_argument('--node_num_g', type=int, default=30, help = 'number of vertices')    
    parser.add_argument('--gcndimension', type=int, default=64, help = 'dimension of latent representation')  
    parser.add_argument('--percentage', type=float, default=0.1, help = 'cutoff')
    parser.add_argument('--num_order', type=int, default=3, help = 'number of simplexes')

    args = parser.parse_args()
    
    
    path=r"../../Database/EEG/"+args.file_name+"/"
    database = args.file_name
    
    series, labels_, classes_len = load_data(path, database)
    
    # GPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('The code uses MPS...')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses CUDA...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')
        
        
    TEST_ACC=[]
    for i in range(args.ie):
        
        # split
        train_series, val_test_series, train_labels_, val_test_labels_ = train_test_split(series, labels_, test_size = 0.4)
        val_series, test_series, val_labels_, test_labels_ = train_test_split(val_test_series, val_test_labels_, test_size = 0.5)
        
        # tdl
        train_seg_g, train_zero_odr_edges, train_cosines, train_one_odr_edges, \
            train_triangle_features, train_two_odr_edges, classes_len, win_size_g, train_sample_num = \
                DATA_g(train_series, args.node_num_g, args.percentage)
        
        val_seg_g, val_zero_odr_edges, val_cosines, val_one_odr_edges, \
            val_triangle_features, val_two_odr_edges, __, __, val_sample_num = \
                DATA_g(val_series, args.node_num_g, args.percentage)
                
        test_seg_g, test_zero_odr_edges, test_cosines, test_one_odr_edges, \
            test_triangle_features, test_two_odr_edges, __, __, test_sample_num \
            = DATA_g(test_series, args.node_num_g, args.percentage)
        
        # multiscale transformer
        train_seg_t1, node_num_t1, train_seg_t2, node_num_t2, train_seg_t3, node_num_t3 = \
            DATA_t(train_series, train_sample_num, args.win_size_t1, args.win_size_t2, args.win_size_t3)
        
        val_seg_t1, __, val_seg_t2, __, val_seg_t3, __ = \
            DATA_t(val_series, val_sample_num, args.win_size_t1, args.win_size_t2, args.win_size_t3)
        
        test_seg_t1, __, test_seg_t2, __, test_seg_t3, __ = \
            DATA_t(test_series, test_sample_num, args.win_size_t1, args.win_size_t2, args.win_size_t3)
    
        # tdl
        loader_train_zero = batch_g(train_seg_g, train_labels_, train_zero_odr_edges, train_sample_num, args.batch_size)
        loader_train_one = batch_g(train_cosines, train_labels_, train_one_odr_edges, train_sample_num, args.batch_size)
        loader_train_two = batch_g(train_triangle_features, train_labels_, train_two_odr_edges, train_sample_num, args.batch_size)
        
        loader_val_zero = batch_g(val_seg_g, val_labels_, val_zero_odr_edges, val_sample_num, args.batch_size)
        loader_val_one = batch_g(val_cosines, val_labels_, val_one_odr_edges, val_sample_num, args.batch_size)
        loader_val_two = batch_g(val_triangle_features, val_labels_, val_two_odr_edges, val_sample_num, args.batch_size)
        
        loader_test_zero = batch_g(test_seg_g, test_labels_, test_zero_odr_edges, test_sample_num, args.batch_size)
        loader_test_one = batch_g(test_cosines, test_labels_, test_one_odr_edges, test_sample_num, args.batch_size)
        loader_test_two = batch_g(test_triangle_features, test_labels_, test_two_odr_edges, test_sample_num, args.batch_size)
        
        # multiscale transformer
        loader_train_t1 = batch_t(train_seg_t1, train_sample_num, args.batch_size)
        loader_train_t2 = batch_t(train_seg_t2, train_sample_num, args.batch_size)
        loader_train_t3 = batch_t(train_seg_t3, train_sample_num, args.batch_size)
        
        loader_val_t1 = batch_t(val_seg_t1, val_sample_num, args.batch_size)
        loader_val_t2 = batch_t(val_seg_t2, val_sample_num, args.batch_size)
        loader_val_t3 = batch_t(val_seg_t3, val_sample_num, args.batch_size)
        
        loader_test_t1 = batch_t(test_seg_t1, test_sample_num, args.batch_size)
        loader_test_t2 = batch_t(test_seg_t2, test_sample_num, args.batch_size)
        loader_test_t3 = batch_t(test_seg_t3, test_sample_num, args.batch_size)
        
            
        # Initialize model, optimizer and loss criterion
        model = HighTS(args, classes_len, win_size_g, node_num_t1, node_num_t2, node_num_t3, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
        criterion = nn.CrossEntropyLoss()
          
        print("==================== BEGIN ====================")
        best_acc = 0
        for epoch in range(args.NUM_EPOCHS):
            # print("==================== TRAIN ====================")
            loss_train , pre_train, true_y_train= train(model, device, loader_train_zero, loader_train_one, 
                                                        loader_train_two, loader_train_t1, loader_train_t2, 
                                                        loader_train_t3, optimizer, classes_len)  
            __, pre_result_train = pre_train.max(dim=1)
            acc_train = accuracy_score(true_y_train.cpu(), pre_result_train.cpu())            
            
            # print("==================== VAL ====================")
            pre_val, true_y_val = predict(model, device, loader_val_zero, loader_val_one, loader_val_two, 
                                            loader_val_t1, loader_val_t2, loader_val_t3, classes_len)
            __, pre_result_val = pre_val.max(dim=1)
            acc_val = accuracy_score(true_y_val.cpu(), pre_result_val.cpu())

            if acc_val > best_acc:
                best_acc = acc_val
                # print("==================== TEST ====================")
                pre_test, true_y_test = predict(model, device, loader_test_zero, loader_test_one, loader_test_two, 
                                                loader_test_t1, loader_test_t2, loader_test_t3, classes_len)
                __, pre_result_test = pre_test.max(dim=1)
                acc_test = accuracy_score(true_y_test.cpu(), pre_result_test.cpu())
                temp_save = torch.cat((true_y_test.cpu().unsqueeze(1), pre_result_test.cpu().unsqueeze(1)),1).numpy()
                save_name = f"{args.file_name}_node={args.node_num_g}_dimG={args.gcndimension}_dimT={args.T_FE_DIM}_{i+1}"
                np.savetxt(r"../../Results/"+args.file_name+"/"+save_name+".csv",temp_save,fmt="%d",delimiter=",",header="Ture,Pre",comments="")
            
            if epoch % 500 == 0:                                                
                print("========================================")
                print(f"Epoch [{epoch + 1}/{args.NUM_EPOCHS}], Loss=",loss_train)
                print("ACC_train=",acc_train)
                print("ACC_val_best=",best_acc)
                print("ACC_test=",acc_test)
                
            if epoch == args.NUM_EPOCHS-1:
                TEST_ACC.append(acc_test)
    
    print("============= RESULTS ==============")
    print("Dataset:",args.file_name)
    print("Node_Num_G:", args.node_num_g)
    print("Dimension:", args.gcndimension)
    print("Best_ACC=",TEST_ACC)
    
    