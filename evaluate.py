# checked
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
from scipy import ndimage
import numpy as np
import sys
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

import jittor as jt
from jittor import nn
from dataset.datasets import CSDataSet, ADE20KDataSet, VOCDataSet
from networks import van, ccnet, deeplabv3, hvnet, ccnet_wo_dsn, ccnet_large

from math import ceil
from PIL import Image as PILImage
from utils.pyt_utils import load_model

from engine import Engine
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = '../data/cityscapes'
DATA_LIST_PATH = './dataset/list/cityscapes/val.lst'
IGNORE_LABEL = 255
BATCH_SIZE = 1
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
INPUT_SIZE = 769
RESTORE_FROM = './snapshots/ccnet_cityscape_TESTCORRECT'
SAVE_PATH = './results/'

ADE20K_IMAGEROOT = '../data/ADEChallengeData2016/images/'
ADE20K_LABELROOT = '../data/ADEChallengeData2016/annotations/'

def get_parser():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--image-dir", type=str, default=ADE20K_IMAGEROOT,
                        help="Path to the directory containing the ADE20K image data.")
    parser.add_argument("--label-dir", type=str, default=ADE20K_LABELROOT,
                        help="Path to the directory containing the ADE20K label data.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--recurrence", type=int, default=2,
                        help="choose the number of recurrence.")

    parser.add_argument("--dataset", type=str, default='cityscape',
                        help="choose the dataset")
    parser.add_argument("--save-path", type=str, default=SAVE_PATH,
                        help="Where to save evaluation results.")

    parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                        help="height and width of images.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="choose the number of recurrence.")
    parser.add_argument("--whole", type=bool, default=False,
                        help="use whole input size.")
    parser.add_argument("--model", type=str, default='ccnet',
                        help="choose model.")
    parser.add_argument("--van-size", type=str, default='van_large',
                        help="choose model size from van_tiny, van_small, van_base, van_large.")
    return parser

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img

def predict_sliding(net, image, tile_size, classes, recurrence):
    # interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)       # TODO: not sure
    image_size = image.shape
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    # print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = np.zeros((image_size[0], image_size[2], image_size[3], classes))
    count_predictions = np.zeros((1, image_size[2], image_size[3], classes))
    tile_counter = 0

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], image_size[3])
            y2 = min(y1 + tile_size[0], image_size[2])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = image[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            
            # exit()
            tile_counter += 1
            # print("Predicting tile %i" % tile_counter)
            padded_prediction = net(jt.Var(padded_img))
            if isinstance(padded_prediction, list):
                padded_prediction = padded_prediction[0]
            # padded_prediction = interp(padded_prediction).numpy().transpose(0,2,3,1)
            padded_prediction = nn.upsample(img=padded_prediction, size=tile_size, mode='bilinear', align_corners=True).numpy().transpose(0,2,3,1)
            prediction = padded_prediction[0, 0:img.shape[2], 0:img.shape[3], :]
            count_predictions[0, y1:y2, x1:x2] += 1
            full_probs[:, y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    # visualize normalization Weights
    # plt.imshow(np.mean(count_predictions, axis=2))
    # plt.show()
    return full_probs

def predict_whole(net, image, tile_size, recurrence):
    N_, C_, H_, W_ = image.shape
    image = jt.Var(image)
    interp = nn.Upsample(size=(H_, W_), mode='bilinear', align_corners=True)        # TODO
    prediction = net(image)
    if isinstance(prediction, list):
        prediction = prediction[0]
    prediction = interp(prediction).numpy().transpose(0,2,3,1)
    return prediction

def predict_multiscale(net, image, tile_size, scales, classes, flip_evaluation, recurrence):
    """
    Predict an image by looking at it with different scales.
        We choose the "predict_whole_img" for the image with less than the original input size,
        for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
    """
    image = image.data
    N_, C_, H_, W_ = image.shape
    full_probs = np.zeros((N_, H_, W_, classes))
    for scale in scales:
        scale = float(scale)
        scale_image = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
        # scaled_probs = predict_whole(net, scale_image, tile_size, recurrence)
        scaled_probs = predict_sliding(net, scale_image, tile_size, classes, recurrence)
        if flip_evaluation == True:
            # flip_scaled_probs = predict_whole(net, scale_image[:,:,:,::-1].copy(), tile_size, recurrence)
            flip_scaled_probs = predict_sliding(net, scale_image[:,:,:,::-1].copy(), tile_size, classes, recurrence)
            scaled_probs = 0.5 * (scaled_probs + flip_scaled_probs[:,::-1,:])
        full_probs += scaled_probs
    full_probs /= len(scales)
    return full_probs

def get_confusion_matrix(gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]
        return confusion_matrix

def main():
    """Create the model and start the evaluation process."""
    parser = get_parser()

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()

        seg_model = eval(args.model + '.Seg_Model')(
            num_classes=args.num_classes,
            img_size=args.input_size,
            recurrence=args.recurrence,
            pretrained=True,
            pretrained_model=args.restore_from,
            van_size=args.van_size
        )
        # print(seg_model)
        
        # load_model(seg_model, args.restore_from)

        model = engine.data_parallel(seg_model)
        model.eval()

        h, w = args.input_size, args.input_size
        input_size = (h, w)
        
        # * can go to this line
        if args.dataset == 'cityscape':
            dataset = CSDataSet(root=args.data_dir, list_path=args.data_list, crop_size=input_size, 
                                    mean=IMG_MEAN, batch_size=1, shuffle=False)
        elif args.dataset == 'ade20k':
            dataset = ADE20KDataSet(image_root=args.image_dir, label_root=args.label_dir, batch_size=1, crop_size=input_size,
                                    train=False, shuffle=False, scale=False, mirror=False)
        elif args.dataset == 'voc':
            dataset = VOCDataSet(root=args.data_dir, list_path=args.data_list, crop_size=input_size,
                                 mean=IMG_MEAN, batch_size=1, shuffle=False)
        test_loader, _ = engine.get_test_loader(dataset)

        data_list = []
        confusion_matrix = np.zeros((args.num_classes,args.num_classes))
        palette = get_palette(256)

        save_path = os.path.join(os.path.dirname(args.save_path), args.restore_from.split(os.sep)[-1], 'outputs')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        # number_of_sample = len(test_loader)
        number_of_sample = 500
        pbar = tqdm(range(number_of_sample), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(test_loader)

        for idx in pbar:
            image, label, size, name = next(dataloader)
            size = size[0].numpy()
            with jt.no_grad():
                output = predict_multiscale(model, image, input_size, [1.0], args.num_classes, False, 0)

            seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
            seg_pred = seg_pred[:, :size[0], :size[1]]
            # print("SSSSSSS", seg_pred.shape)
            seg_gt = np.asarray(label.numpy()[:,:size[0],:size[1]], dtype=np.uint8)
            # print("SSSSSSS", seg_gt.shape)
            
            for i in range(image.size(0)):
                output_im = PILImage.fromarray(seg_pred[i])
                output_im.putpalette(palette)
                output_im.save(os.path.join(save_path, name[i]+'_output.png'))
                # print(output_im.size)
                
                output_lb = PILImage.fromarray(seg_gt[i])
                output_lb.putpalette(palette)
                output_lb.save(os.path.join(save_path, name[i]+'_label.png'))
                # print(output_lb.size)
                
            ignore_index = seg_gt != args.ignore_label
            ignore_index = jt.Var(ignore_index)
            seg_gt = jt.Var(seg_gt)
            seg_pred = jt.Var(seg_pred)
            
            seg_gt = seg_gt[ignore_index]
            seg_pred = seg_pred[ignore_index]
            # exit()
            
            # show_all(gt, output)
            confusion_matrix += get_confusion_matrix(np.asarray(seg_gt.numpy(), dtype=int), np.asarray(seg_pred.numpy(), dtype=int), args.num_classes)

            print_str = ' Iter{}/{}'.format(idx + 1, number_of_sample)
            pbar.set_description(print_str, refresh=False)

        # confusion_matrix = jt.Var(confusion_matrix)
        # confusion_matrix = engine.all_reduce_tensor(confusion_matrix, norm=False).numpy()
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)

        IU_array = (tp / np.maximum(1.0, pos + res - tp))[1:]
        mean_IU = IU_array.mean()
        
        # getConfusionMatrixPlot(confusion_matrix)
        # if engine.distributed and engine.local_rank == 0:
        print({'meanIU':mean_IU, 'IU_array':IU_array})
        model_path = os.path.join(os.path.dirname(args.save_path), args.restore_from.split(os.sep)[-1])
        with open(os.path.join(model_path, 'result.txt'), 'w') as f:
            f.write(json.dumps({'meanIU':mean_IU, 'IU_array':IU_array.tolist(), 'number_of_sample':number_of_sample}))

if __name__ == '__main__':
    main()
