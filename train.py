# unchecked
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

import json
import jittor as jt
import numpy as np
import jittor.optim as optim
import sys
from tqdm import tqdm
import os.path as osp

from dataset.datasets import ADE20KDataSet, CSDataSet
from networks import van, ccnet, deeplabv3, hvnet, ccnet_wo_rcca, ccnet_large


from loss.criterion import CriterionDSN, CriterionOhemDSN
from engine import Engine
# from utils.encoding import DataParallelModel, DataParallelCriterion


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

MODEL = 'ccnet'
BATCH_SIZE = 6
DATA_DIRECTORY = '../data/cityscapes'  # unused while using ADEChallenge
DATA_LIST_PATH = './dataset/list/cityscapes/train.lst'  # unused while using ADEChallenge
IGNORE_LABEL = 255
INPUT_SIZE = 769
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 10000
POWER = 0.9
RANDOM_SEED = 12345
RESTORE_FROM = './dataset/resnet101-imagenet.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 500  # * changed
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
VAN_SIZE = 'van_base'

# ADE20K_IMAGEROOT = '/mfs/xueshuxinxing-jz/siyuan/data/ADEChallengeData2016/images/'
# ADE20K_LABELROOT = '/mfs/xueshuxinxing-jz/siyuan/data/ADEChallengeData2016/annotations/'
ADE20K_IMAGEROOT = '../data/ADEChallengeData2016/images/'
ADE20K_LABELROOT = '../data/ADEChallengeData2016/annotations/'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--image-dir", type=str, default=ADE20K_IMAGEROOT,
                        help="Path to the directory containing the ADE20K image data.")
    parser.add_argument("--label-dir", type=str, default=ADE20K_LABELROOT,
                        help="Path to the directory containing the ADE20K label data.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                        help="height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--print-frequency", type=int, default=50,
                        help="Number of training steps.") 
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=str, default='None',
                        help="choose gpu device.")
    parser.add_argument("--model", type=str, default=MODEL, 
                        help='choose model.')
    parser.add_argument("--van-size", type=str, default='van_base',
                        help="choose model size from van_tiny, van_small, van_base, van_large.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="choose the number of workers.")
    parser.add_argument("--recurrence", type=int, default=2,
                        help="choose the number of recurrence.")
    
    parser.add_argument("--dataset", type=str, default='cityscape',
                        help="choose the dataset")

    parser.add_argument("--ohem", type=str2bool, default='False',
                        help="use hard negative mining")
    parser.add_argument("--ohem-thres", type=float, default=0.6,
                        help="choose the samples with correct probability underthe threshold.")
    parser.add_argument("--ohem-keep", type=int, default=200000,
                        help="choose the samples with correct probability underthe threshold.")
    return parser

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(optimizer, learning_rate, i_iter, max_iter, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(learning_rate, i_iter, max_iter, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

# def set_bn_eval(m):
#     classname = m.__class__.__name__
#     if classname.find('BatchNorm') != -1:
#         m.eval()

# def set_bn_momentum(m):
#     classname = m.__class__.__name__
#     if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
#         m.momentum = 0.0003

def main():
    """Create the model and start the training."""
    parser = get_parser()
    # print('======== Test Cuda ========')
    # print(jt.has_cuda)
    jt.flags.use_cuda = jt.has_cuda
    # exit(0)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

        # cudnn.benchmark = True
        seed = args.random_seed
        # if engine.distributed:
        #     seed = engine.local_rank
        jt.set_global_seed(seed)

        # data loader
        h, w = args.input_size, args.input_size
        input_size = (h, w)

        if args.dataset == 'cityscape':
            dataset = CSDataSet(root=args.data_dir, list_path=args.data_list, max_iters=480000, crop_size=input_size, 
                                scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN, batch_size=args.batch_size, shuffle=True)
        elif args.dataset == 'ade20k':
            dataset = ADE20KDataSet(image_root=args.image_dir, label_root=args.label_dir, batch_size=args.batch_size, crop_size=input_size,
                                    scale=args.random_scale, mirror=args.random_mirror, max_iters=480000, shuffle=True)
        train_loader, _ = engine.get_train_loader(dataset)
        # config network and criterion
        if args.ohem:
            criterion = CriterionOhemDSN(thresh=args.ohem_thres, min_kept=args.ohem_keep)
        else:
            criterion = CriterionDSN() #CriterionCrossEntropy()

        # model = Res_Deeplab(args.num_classes, criterion=criterion,
        #         pretrained_model=args.restore_from)
        print("-------Now training with {} model-------".format(args.model))
        print(args.restore_from)
        model = eval(args.model + '.Seg_Model')(
            input_size=args.input_size,
            num_classes=args.num_classes,
            recurrence=args.recurrence,
            pretrained_model=args.restore_from,
            van_size=args.van_size
        )
        # model = seg_model
        # model.init_weights()
        # print(seg_model)
        # exit(0)

        # group weight and config optimizer
        # optimizer = optim.SGD([{'params': filter(lambda p: p.requires_grad, seg_model.parameters()), 'lr': args.learning_rate}], 
        #         lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
        optimizer.zero_grad()

        print('-------Start Training-------')
        model.train()
        # jt.sync_all(True)

        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        run = True
        global_iteration = args.start_iters
        all_loss = []
        
        save_model_at = 'R' + str(args.recurrence) + '_' + args.dataset
        if args.model == 'van':
            save_model_at = args.van_size + '_' + save_model_at
        else:
            save_model_at = args.model + '_' + save_model_at

        while run:
            epoch = global_iteration // len(train_loader)
            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
            pbar = tqdm(range(len(train_loader)), file=sys.stdout,
                        bar_format=bar_format)
            dataloader = iter(train_loader)
            # print('Trainloader len: {}'.format(len(train_loader)))
            # print('-------Dataloader prepared-------')

            for idx in pbar:
                global_iteration += 1

                images, labels, _, _ = next(dataloader)
                labels = labels.long()

                lr = adjust_learning_rate(optimizer, args.learning_rate, global_iteration-1, args.num_steps, args.power)
                # print('--------Feeding Data--------')
                outs = model(images, labels)
                # print('--------Data Comes Out--------')

                # print(outs[0].shape, labels.shape)
                # exit()
                loss = criterion(outs, labels)
                # print('--------Reducing Data--------')
                loss = engine.all_reduce_tensor(loss)  # ! Is this line necessary?
                # print('--------Data Reduced--------')
                optimizer.step(loss)
                # print('--------Optimizer Steped--------')


                print_str = 'Epoch{}/Iters{}'.format(epoch, global_iteration) \
                        + ' Iter{}/{}:'.format(idx + 1, len(train_loader)) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % loss.item()  # * changed
                all_loss.append(loss.item())

                pbar.set_description(print_str, refresh=False)

                if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                    if global_iteration % args.save_pred_every == 0 or global_iteration >= args.num_steps:
                        print('taking snapshot ...')
                        jt.save(model.state_dict(), osp.join(args.snapshot_dir, save_model_at + '_' + str(global_iteration)))
                        with open(osp.join(args.snapshot_dir, save_model_at + '_' + str(global_iteration) + '_result.txt'), 'w') as f:
                            f.write(json.dumps({'all_loss': all_loss, 'batch_size': args.batch_size, 
                                                'total_iteration': global_iteration, 'image_size': args.input_size}))

                if global_iteration >= args.num_steps:
                    run = False
                    break
        

if __name__ == '__main__':
    main()
