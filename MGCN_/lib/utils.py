import os
import numpy as np
import torch
import torch.utils.data


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    A[j, i] = 1
                    distaneA[i, j] = distance
                    distaneA[j, i] = distance
            return A, distaneA



def compute_val_loss_mgcn(net, val_loader, criterion,  masked_flag,missing_value,sw, epoch, limit=None):

    DEVICE = torch.device('cuda:0')
    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        for batch_index, (encoder_inputs, labels ) in enumerate(val_loader):

            encoder_inputs=encoder_inputs.float().to(DEVICE)
            labels = labels.float().to(DEVICE)
            outputs = net(encoder_inputs)
            if masked_flag:
                loss = criterion(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)

            tmp.append(loss.item())
            if batch_index % 50 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss



def predict_and_save_results_mgcn(net, data_loader,test_data, pred_len,global_step, metric_method, params_path, type):

    DEVICE = torch.device('cuda:0')
    net.train(False)  # ensure dropout layers are in test mode
    preds = []
    trues = []
    with torch.no_grad():

        loader_length = len(data_loader)  # nb of batch

        input = []  # 存储所有batch的input

        for batch_index, (encoder_inputs, labels) in enumerate(data_loader):

            input.append(encoder_inputs[:, :, 0:1].cpu().numpy())  # (batch, T', 1)
            encoder_inputs=encoder_inputs.float().to(DEVICE)
            labels = labels.float().to(DEVICE)
            outputs = net(encoder_inputs)

            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            pred = outputs
            true = labels

            preds.append(pred)
            trues.append(true)

            if batch_index % 100 == 0:
                print('predicting dataset set batch %s / %s' % (batch_index + 1, loader_length))

    preds = np.array(preds)
    trues = np.array(trues)
    print('test shape:', preds.shape, trues.shape)

    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    mae, mse, rmse, mape = metric(preds, trues)
    print('rmse:{},mse:{},mae:{}'.format(rmse, mse, mae))

    return

def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)

    return mae, mse, rmse, mape

def adjust_learning_rate(optimizer, epoch, learning_rate):

    lr_adjust = {epoch: learning_rate * (1 ** ((epoch - 1) // 2))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, net, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            torch.save(net.state_dict(), path)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            torch.save(net.state_dict(), path)
            print("save model")
            self.counter = 0

