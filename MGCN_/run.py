import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import shutil
import argparse
import configparser
from model.MGCN import make_model
from lib.utils import get_adjacency_matrix, compute_val_loss_mgcn, predict_and_save_results_mgcn
from tensorboardX import SummaryWriter
from lib.metrics import masked_mape_np, masked_mae,masked_mse,masked_rmse
from data_provider.data_factory import data_provider
import random
from lib.utils import EarlyStopping, adjust_learning_rate


parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMS04_mgcn.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']
model_name = training_config['model_name']

ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE)

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
nb_chev_filter = int(training_config['nb_chev_filter'])
nb_time_filter = int(training_config['nb_time_filter'])
in_channels = int(training_config['in_channels'])
K = int(training_config['K'])
loss_function = training_config['loss_function']
metric_method = training_config['metric_method']
missing_value = float(training_config['missing_value'])
time_strides=1

folder_dir = 'predict%s_MGCN' % (num_for_predict)
print('folder_dir:', folder_dir)
params_path = os.path.join('experiments', dataset_name, folder_dir)
print('params_path:', params_path)


def _get_data(root_path, flag,seq_len,label_len,pred_len,batch_size):
    data_set, data_loader = data_provider(root_path, flag,seq_len,label_len,pred_len,batch_size)
    return data_set, data_loader
train_data, train_loader = _get_data(root_path=graph_signal_matrix_filename,flag='train',seq_len=points_per_hour,label_len=0,pred_len=num_for_predict,batch_size=batch_size)
vali_data, val_loader = _get_data(root_path=graph_signal_matrix_filename,flag='val',seq_len=points_per_hour,label_len=0,pred_len=num_for_predict,batch_size=batch_size)
test_data, test_loader = _get_data(root_path=graph_signal_matrix_filename,flag='test',seq_len=points_per_hour,label_len=0,pred_len=num_for_predict,batch_size=batch_size)


adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)

net = make_model(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx,
                 num_for_predict, len_input)


def train_main():
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')


    early_stopping = EarlyStopping(patience=5)

    print('param list:')
    print('CUDA\t', DEVICE)
    print('in_channels\t', in_channels)
    print('nb_chev_filter\t', nb_chev_filter)
    print('nb_time_filter\t', nb_time_filter)
    print('batch_size\t', batch_size)
    print('graph_signal_matrix_filename\t', graph_signal_matrix_filename)
    print('start_epoch\t', start_epoch)
    print('epochs\t', epochs)
    masked_flag=0
    criterion = nn.L1Loss().to(DEVICE)
    criterion_masked = masked_mae
    if loss_function=='masked_mse':
        criterion_masked = masked_mse         #nn.MSELoss().to(DEVICE)
        masked_flag=1
    elif loss_function=='masked_mae':
        criterion_masked = masked_mae
        masked_flag = 1
    elif loss_function == 'mae':
        criterion = nn.L1Loss().to(DEVICE)
        masked_flag = 0
    elif loss_function == 'rmse':
        criterion = nn.MSELoss().to(DEVICE)
        masked_flag= 0
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    sw = SummaryWriter(logdir=params_path, flush_secs=3)
    print(net)

    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size())
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    time_now = time.time()
    if start_epoch > 0:

        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)

        net.load_state_dict(torch.load(params_filename))

        print('start epoch:', start_epoch)

        print('load weight from: ', params_filename)

    # train model
    for epoch in range(start_epoch, epochs):
        iter_count = 0
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        if masked_flag:
            val_loss = compute_val_loss_mgcn(net, val_loader, criterion_masked, masked_flag,missing_value,sw, epoch,DEVICE)
        else:
            val_loss = compute_val_loss_mgcn(net, val_loader, criterion, masked_flag, missing_value, sw, epoch,DEVICE)


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        
        early_stopping(val_loss, net, params_filename)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        adjust_learning_rate(optimizer, epoch + 1, learning_rate)
      
        net.train()  # ensure dropout layers are in train mode

        for batch_index, (encoder_inputs, labels) in enumerate(train_loader):
            iter_count += 1

            encoder_inputs=encoder_inputs.float().to(DEVICE)
            labels = labels.float().to(DEVICE)
            optimizer.zero_grad()

            outputs = net(encoder_inputs)

            if masked_flag:
                loss = criterion_masked(outputs, labels,missing_value)
            else :
                loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)

            if (batch_index + 1) % 300 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(batch_index + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                print(speed)
                allocated_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
                cached_memory = torch.cuda.memory_cached() / (1024 * 1024 * 1024)
                total = allocated_memory + cached_memory
                print('allocated_memory:', allocated_memory)
                print('cached_memory:', cached_memory)
                print('total:', total)
                iter_count = 0
                time_now = time.time()

    print('best epoch:', best_epoch)

    # apply the best model on the test set
    predict_main(best_epoch, test_loader,test_data,num_for_predict,metric_method , 'test')

def predict_main(global_step, data_loader,test_data, pred_len,metric_method, type):

    params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    print('load weight from:', params_filename)

    net.load_state_dict(torch.load(params_filename))

    predict_and_save_results_mgcn(net, data_loader, test_data, pred_len,global_step, metric_method, params_path, type)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    fix_seed = 2024
    set_seed(fix_seed)
    train_main()

