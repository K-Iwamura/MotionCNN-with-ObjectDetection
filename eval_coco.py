# We heavily borrow code from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils_motion_caption import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import pickle

import sys
sys.path.insert(1, 'coco-caption')
#from pycocoevalcap.bleu.bleu import Bleu
#from pycocoevalcap.rouge.rouge import Rouge
#from pycocoevalcap.cider.cider import Cider
#from pycocoevalcap.meteor.meteor import Meteor
#from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
#import json
#import collections as cl



# Parameters
data_folder = '/home/Iwamura/Workspace/caption_motion/datasets_temp/final_dataset'
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint = ''

word_map_file = ''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder_opt = checkpoint['encoder_opt']
encoder_opt = encoder_opt.to(device)
encoder_opt.eval()



# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization class
    
normalize_opt = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])



def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize_opt])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()
    
    # For each image
    for i, (image_opt, image_bottom, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size
        count_num = i
        image_opt = image_opt
        image_bottomup = image_bottom.to(device)
        image_opt = image_opt.to(device)

        image_opt = encoder_opt(image_opt)

        

        # Encode

        encoder_out_opt = image_opt
        bottomup_feat = image_bottomup
        bottomup_feat_mean = image_bottomup.mean(1)
        enc_image_size = encoder_out_opt.size(1)
        encoder_dim = encoder_out_opt.size(3)

        # Flatten encoding

        encoder_out_opt = encoder_out_opt.view(1, -1, encoder_dim)
        bottomup_feat = bottomup_feat.view(1, -1, encoder_dim)
        num_pixels = encoder_out_opt.size(1)
        num_pixels_bottomup = bottomup_feat.size(1)
        num_pixels_bottomup_mean= bottomup_feat_mean.size(1)

        # We'll treat the problem as having a batch size of k
       # encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)# (k, num_pixels, encoder_dim)
        encoder_out_opt = encoder_out_opt.expand(k, num_pixels, encoder_dim)
        
        #bottomup_feat = bottomup_feat.expand(k, num_pixels_bottomup, encoder_dim)
        bottomup_feat_mean = bottomup_feat_mean.expand(k, encoder_dim)


               # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h1, c1 = decoder.init_hidden_state(bottomup_feat_mean)
        h2, c2 = decoder.init_hidden_state(bottomup_feat_mean)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            h1, c1 = decoder.decode_step(
                torch.cat([h2, bottomup_feat_mean, embeddings], dim=1), (h1, c1))

            awe_opt, alpha_opt = decoder.attention(encoder_out_opt, h1)  # (s, encoder_dim), (s, num_pixels)

            aweb, alpha_bottomup = decoder.attention_bottomup(bottomup_feat, h1)
            
            gate_opt = decoder.sigmoid(decoder.f_beta(h1))  # gating scalar, (s, encoder_dim)

            gated_awe_opt = gate_opt * awe_opt

            gated_awe_cat = torch.cat([gated_awe_opt, aweb], dim=1)

            h2, c2 = decoder.language_model(
                torch.cat([gated_awe_cat, h1], dim=1), (h2, c2))  # (s, decoder_dim)

            scores = decoder.fc(h2)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h1 = h1[prev_word_inds[incomplete_inds]]
            c1 = c1[prev_word_inds[incomplete_inds]]
            h2 = h2[prev_word_inds[incomplete_inds]]
            c2 = c2[prev_word_inds[incomplete_inds]]
            bottomup_feat_mean = bottomup_feat_mean[prev_word_inds[incomplete_inds]]
            encoder_out_opt = encoder_out_opt[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 18:
                break
            step += 1

            
        
        if len(complete_seqs_scores):
            i = complete_seqs_scores.index(max(complete_seqs_scores))
        else:
            continue
            

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)


    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    # hypo_dict: {id: [{id: sentence}], ...}
    ref_dict  = {}
    hypo_dict = {}

    # Calculate BLEU, METEOR, ROUGE_L, CIDEr scores
    for idx_ref, refs in enumerate(references):
        # 5sentences list
        sent5_list = []
#        print("refs = {}".format(refs))
        # refs: 5sentences to 1 image
        for ref in refs:
            # ref: 1 captons

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
    with open('./results/ref_dict_Version00_checkpoint_coco_5_cap.pickle', mode='wb') as f:
        pickle.dump(ref_dict, f, protocol=2)
    with open('./results/hypo_dict_Version00_checkpoint_coco_5_cap.pickle', mode='wb') as g:
        pickle.dump(hypo_dict, g, protocol=2)


    
    return bleu4


if __name__ == '__main__':
    beam_size = 3
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))

