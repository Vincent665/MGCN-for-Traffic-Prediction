from data_provider.data_loader import Dataset_PEMS
from torch.utils.data import DataLoader

data_dict = {
    'PEMS': Dataset_PEMS,
}


def data_provider(root_path, flag,seq_len,label_len,pred_len,batch_size):
    Data = data_dict['PEMS']

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation

    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = batch_size  # bsz for train and valid


    data_set = Data(
        root_path=root_path,
        flag=flag,
        size=[seq_len, label_len, pred_len],

    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=10,
        drop_last=drop_last)
    return data_set, data_loader
