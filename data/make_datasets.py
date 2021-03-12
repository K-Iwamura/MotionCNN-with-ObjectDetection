import os
import numpy as np
import h5py
import json
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample

import cv2
import torchvision
import math
import argparse
from predictor import COCODemo
from maskrcnn_benchmark.config import cfg
#from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import pickle

parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
parser.add_argument(
        "--config-file",
        default="~/maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
)
parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.2,
        help="Minimum score for the prediction to be shown",
)
parser.add_argument(
        "--min-image-size",
        type=int,
        default=256,#224
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
)
parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
)
parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=6,
        help="Number of heatmaps per dimension to show",
)
parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
)

args = parser.parse_args()
# load config from file and command-line arguments
cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

mscoco_maskRCNN = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,

    )
cfg.freeze()

def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=18):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    with open(os.path.join(output_folder,'train36_imgid2idx.pkl'), 'rb') as j:
        train_data = pickle.load(j)

    with open(os.path.join(output_folder,'val36_imgid2idx.pkl'), 'rb') as j:
        val_data = pickle.load(j)
        

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    train_opt_paths = []
    val_image_paths = []
    val_image_captions = []
    val_opt_paths = []
    test_image_paths = []
    test_image_captions = []
    test_opt_paths = []

    train_image_det = []
    val_image_det = []
    test_image_det = []
    word_freq = Counter()

    for break_num, img in enumerate(tqdm(data['images'])):
        captions = []

        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])

            c_tmp = c['tokens']

            captions.append(c_tmp[:max_len])


        if len(captions) == 0:
            continue

        image_id = img['filename'].split('_')[2]
        image_id = int(image_id.lstrip("0").split('.')[0])
        

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        opt_path = os.path.join(image_folder + '/opticalflow_Im2Flow/', img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(image_folder + '/opticalflow_Im2Flow/',img['filename'])

        if img['split'] in {'train', 'restval'}:
            if img['filepath'] == 'train2014':
                if image_id in train_data:
                    train_image_det.append(("t",train_data[image_id]))
            else:
                if image_id in val_data:
                    train_image_det.append(("v",val_data[image_id]))
            
            train_image_paths.append(path)
            train_image_captions.append(captions)
            train_opt_paths.append(opt_path)
            
        elif img['split'] in {'val'}:
            if image_id in val_data:
                val_image_det.append(("v",val_data[image_id]))
            val_image_paths.append(path)
            val_image_captions.append(captions)
            val_opt_paths.append(opt_path)
            
        elif img['split'] in {'test'}:
            if image_id in val_data:
                test_image_det.append(("v",val_data[image_id]))
            test_image_paths.append(path)
            test_image_captions.append(captions)
            test_opt_paths.append(opt_path)
      
    
    print("train img{}, cap{}, opt{}".format(len(train_image_paths), len(train_image_captions), len(train_opt_paths)))
    print("val img{}, cap{}, opt{}".format(len(val_image_paths), len(val_image_captions), len(val_opt_paths)))
    print("test img{}, cap{}, opt{}".format(len(test_image_paths), len(test_image_captions), len(test_opt_paths)))

    print("det train{}, val{}, test{}".format(len(train_image_det), len(val_image_det), len(test_image_det)))
            
    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0
    print("words size = {}".format(len(words)))

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths_det, impaths, optpaths, imcaps, split in [(train_image_det, train_image_paths, train_opt_paths, train_image_captions, 'TRAIN'),
                                   (val_image_det, val_image_paths, val_opt_paths, val_image_captions, 'VAL'),
                                   (test_image_det, test_image_paths, test_opt_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')
            opts = h.create_dataset('opticalflow',(len(optpaths), 3, 256, 256), dtype='uint8')

            
            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, (path_det, path, optpath) in enumerate(tqdm(zip(impaths_det, impaths, optpaths))):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255
                # Object detection (get bounding box)
                img_opencv = img.transpose(1, 2, 0)
                img_opencv = cv2.cvtColor(img_opencv, cv2.COLOR_RGB2BGR)
                result, boxes, labels = mscoco_maskRCNN.run_on_opencv_image(img_opencv)

                 # Read opts
                opt = imread(optpaths[i])
                if len(opt.shape) == 2:
                    opt = opt[:, :, np.newaxis]
                    opt = np.concatenate([opt, opt, opt], axis=2)
                opt = imresize(opt, (256, 256))
                opt = opt.transpose(2, 0, 1)
                assert opt.shape == (3, 256, 256)
                assert np.max(opt) <= 255

                # opticalflow crop with bounding box
                opt_obj = np.random.randn(3, 256, 256) * 5e-9
                for box in boxes:
                    box_np = box.numpy()
                    opt_obj[:, int(box_np[1]):int(box_np[3]), int(box_np[0]):int(box_np[2])] =\
                                opt[:, int(box_np[1]):int(box_np[3]), int(box_np[0]):int(box_np[2])]

                # Save image to HDF5 file
                images[i] = img
                if len(boxes)==0:
                    if i <40:
                        print("opt")
                    opts[i] = opt
                else:
                    if i< 40:
                        print("opt_obj")
                    opts[i] = opt_obj
            

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

        # Save bottom up features indexing to JSON files
        with open(os.path.join(output_folder, 'TRAIN' + '_GENOME_DETS_' + base_filename + '.json'), 'w') as j:
            json.dump(train_image_det, j)

        with open(os.path.join(output_folder, 'VAL' + '_GENOME_DETS_' + base_filename + '.json'), 'w') as j:
            json.dump(val_image_det, j)

        with open(os.path.join(output_folder, 'TEST' + '_GENOME_DETS_' + base_filename + '.json'), 'w') as j:
            json.dump(test_image_det, j)



def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'useAll_32batch_img_checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
