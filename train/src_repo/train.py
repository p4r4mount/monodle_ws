import torch
import numpy as np
from torch.utils.data import DataLoader
from Rope_dataset import Rope_Dataset
from model.centernet3d import CenterNet3D
from optimizer import build_optimizer
from scheduler import build_lr_scheduler
from tqdm import tqdm
from losses.centernet_loss import compute_centernet3d_loss
import warnings
import os
warnings.filterwarnings("ignore")

# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def train(max_epoch=200, save_fre=50, save_pth='../../../models'):
    start_epoch = 0
    now_epoch = 0
    progress_bar = tqdm(range(start_epoch, max_epoch), dynamic_ncols=True, leave=True, desc='epochs')
    for epoch in range(start_epoch, max_epoch):
        np.random.seed(np.random.get_state()[1][0] + epoch)
        train_one_epoch(now_epoch,max_epoch)
        now_epoch += 1

        # 更新学习率
        if warmup_lr_scheduler is not None and epoch < 5:
            warmup_lr_scheduler.step()
        else:
            lr_scheduler.step()

        if (now_epoch % save_fre) == 0:
            # ckpt_pth = '/project/train/models'
            ckpt_pth = save_pth
            ckpt_name = os.path.join(ckpt_pth, 'checkpoint_epoch_%d' % now_epoch)
            save_checkpoint(get_checkpoint_state(model, optimizer, now_epoch), ckpt_name)

        progress_bar.update()
    return None
            


def train_one_epoch(now_epoch,max_epoch):
    model.train()
    progress_bar = tqdm(total=len(train_loader), leave=(now_epoch+1 == max_epoch), desc='iters')
    for batch_idx, (inputs, targets, _) in enumerate(train_loader):
        inputs = inputs.to(device)
        for key in targets.keys():
            targets[key] = targets[key].to(device)

        # train one batch
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(targets['size_2d'].sum())
        total_loss, stats_batch = compute_centernet3d_loss(outputs, targets)
        # print(total_loss,'\n')
        # print(stats_batch)
        # if torch.isnan(total_loss):
        #     continue
        total_loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=total_loss)
        progress_bar.update()
    progress_bar.close()


def get_checkpoint_state(model=None, optimizer=None, epoch=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state}


def save_checkpoint(state, filename):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description='monodle for Rope3D')
    parser.add_argument('--bs', dest='batch_size', default=4)
    parser.add_argument('--downsample', dest='downsample', default=8)
    parser.add_argument('--epochs', dest='max_epoch', default=140)
    parser.add_argument('--save_fre', dest='save_fre', default=70)
    parser.add_argument('--depth_threshold', dest='depth_threshold', default=120)
    parser.add_argument('--save_pth', dest='save_pth', default='/home/song/Proj/Rope3D/monodle_models')
    args = parser.parse_args()

    batch_size = args.batch_size
    workers = 1
    downsample = args.downsample
    max_epoch = args.max_epoch
    max_epoch = args.max_epoch
    depth_threshold = args.depth_threshold
    save_pth = args.save_pth
    # save_pth = '/project/train/models'

    # perpare dataset
    train_set = Rope_Dataset(mode='train',downsample=downsample,depth_threshold=depth_threshold)
    test_set = Rope_Dataset(mode='val',downsample=downsample,depth_threshold=depth_threshold)

    # prepare dataloader
    train_loader = DataLoader(dataset=train_set,
                                batch_size=batch_size,
                                num_workers=workers,
                                worker_init_fn=my_worker_init_fn,
                                shuffle=True,
                                pin_memory=False,
                                drop_last=True)
    test_loader = DataLoader(dataset=test_set,
                                batch_size=batch_size,
                                num_workers=workers,
                                worker_init_fn=my_worker_init_fn,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False)
    print('load data finished!')

    # 模型
    model = CenterNet3D(backbone='dla34', neck='DLAUp', num_class=train_set.num_classes, downsample=downsample)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model, device_ids=[0]).to(device)
    print('load model finished!')
    # 优化器
    optim_cfg = {
        'type':'adam',
        'lr':0.00125,
        'weight_decay': 0.00001
    }
    optimizer = build_optimizer(optim_cfg, model)
    # 学习策略
    lr_cfg = {
        'warmup':True,  # 5 epoches, cosine warmup, init_lir=0.00001 in default
        'decay_rate':0.1,
        'decay_list':[90, 120]
    }
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(lr_cfg, optimizer, last_epoch=-1)

    # 训练！
    train(max_epoch,save_fre = 50,save_pth=save_pth)