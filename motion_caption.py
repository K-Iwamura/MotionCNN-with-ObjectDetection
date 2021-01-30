# We heavily borrow code from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils_motion_caption import *
from nltk.translate.bleu_score import corpus_bleu
import sys
from PIL import Image
import h5py
import random
import numpy as np
import torch
from pycocoevalcap3.cider.cider import Cider
from pycocoevalcap3.bleu.bleu import Bleu
from pycocoevalcap3.rouge.rouge import Rouge
from pycocoevalcap3.meteor.meteor import Meteor
from pycocoevalcap3.tokenizer.ptbtokenizer import PTBTokenizer

# Data parameters
data_folder = '/home/Iwamura/Workspace/caption_motion/'

data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 300  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = False # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 16
workers = 0  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-5  # learning rate for encoder if fine-tuning
encoder_opt_lr = 1e-4
decoder_lr = 1e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False # fine-tune encoder?
fine_tune_encoder_opt = False

checkpoint = None


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, rev_word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    rev_word_map = {v: k for k, v in word_map.items()}

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        pretrained_embs, pretrained_embs_dim = load_embeddings('/home/Iwamura/datasets/datasets/GloVe/glove.6B.300d.txt', word_map)
        assert pretrained_embs_dim == decoder.embed_dim
        decoder.load_pretrained_embeddings(pretrained_embs)
        decoder.fine_tune_embeddings(True)

        
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder_opt = Encoder()    
        encoder.fine_tune(fine_tune_encoder)
        encoder_opt.fine_tune(fine_tune_encoder_opt)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
        encoder_optimizer_opt = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_opt.parameters()),
                                                 lr=encoder_opt_lr) if fine_tune_encoder_opt else None
    
    else:

        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder_opt = checkpoint['encoder_opt']
        encoder_optimizer_opt = checkpoint['encoder_optimizer_opt']

        


       # if fine_tune_encoder is True and encoder_optimizer is None and encoder_optimizer_opt is None
        if fine_tune_encoder_opt is True and encoder_optimizer_opt is None:
            encoder_opt.fine_tune(fine_tune_encoder_opt)

            encoder_optimizer_opt = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_opt.parameters()),
                                                 lr=encoder_opt_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)

    encoder_opt = encoder_opt.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders

    normalize_opt = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize_opt])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize_opt])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 10:
            break
        if epoch > 0 and epoch % 4 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)

            if fine_tune_encoder_opt:
                adjust_learning_rate(encoder_optimizer_opt, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder_opt=encoder_opt,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer_opt=encoder_optimizer_opt,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder_opt=encoder_opt,
                                decoder=decoder,
                                criterion=criterion)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder_opt, decoder,
                        encoder_optimizer_opt, decoder_optimizer, recent_bleu4, is_best)
    


def train(train_loader, encoder_opt, decoder, criterion, encoder_optimizer_opt, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder_opt.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, imgs_bottomup, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)
        
      
        # Move to GPU, if available

        imgs_opt = imgs

        imgs_bottomup = imgs_bottomup.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)


        imgs_opt = imgs_opt.to(device)

        # Forward prop
        imgs_opt = encoder_opt(imgs_opt)

        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs_opt, imgs_bottomup, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        
        # Calculate loss
        loss = criterion(scores[0], targets[0])


        # Back prop.
        decoder_optimizer.zero_grad()

        if encoder_optimizer_opt is not None:
            encoder_optimizer_opt.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)

            if encoder_optimizer_opt is not None:
                clip_gradient(encoder_optimizer_opt, grad_clip)

        # Update weights
        decoder_optimizer.step()

        if encoder_optimizer_opt is not None:
            encoder_optimizer_opt.step()

        # Keep track of metrics
        top5 = accuracy(scores[0], targets[0], 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder_opt, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder_opt is not None:
      #  encoder.eval()
        encoder_opt.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    global rev_word_map

    with torch.no_grad():
        # Batches
        for i, (imgs, imgs_bottomup, caps, caplens, allcaps) in enumerate(val_loader):
            # Move to device, if available

            imgs_opt = imgs

            imgs_bottomup = imgs_bottomup.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            imgs_opt = imgs_opt.to(device)

            # Forward prop.
            if encoder_opt is not None:

                imgs_opt = encoder_opt(imgs_opt)
       
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs_opt, imgs_bottomup, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores[0], targets[0])


            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores[0], targets[0], 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)
            
        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)
        ref_dict = {}
        hypo_dict = {}

        for idx_ref, refs in enumerate(references):
            sent5_list = []
            for ref in refs:
                word_sentence = ','.join([rev_word_map[ref_] for ref_ in ref])
                word_sentence = word_sentence.replace(',', ' ')
                sent5_list.append(word_sentence)
            ref_dict[idx_ref] = sent5_list

        for idx_hypo, hypo in enumerate(hypotheses):
            sent1_list = []
            hypo_word_sentence = ','.join([rev_word_map[hypo_] for hypo_ in hypo])
            hypo_word_sentence = hypo_word_sentence.replace(',', ' ')
            sent1_list.append(hypo_word_sentence)
            hypo_dict[idx_hypo] = sent1_list

        
        scorers =[
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
            ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref_dict, hypo_dict)
            if type(score)==list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score        

        print("final scores = {}".format(final_scores))
        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return final_scores["CIDEr"]#bleu4

            

if __name__ == '__main__':
    seed_torch()
    main()
