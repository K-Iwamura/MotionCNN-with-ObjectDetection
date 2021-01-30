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
data_folder = ''
data_name = ''  # base name shared by data files
checkpoint = ''

word_map_file = ''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()
encoder_opt = checkpoint['encoder_opt']
encoder_opt = encoder_opt.to(device)
encoder_opt.eval()



# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization class
class NormalizeWithName(object):
    def __init__(self, mean, std):
        self.mean = list(map(float, mean))
        self.std = list(map(float, std))                                                         

    def __call__(self, data): # tensor                                                                  
        # TODO: make efficient                                                                         
        tensor = torch.FloatTensor(data['image'])

        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s).float()

        return {'image': tensor.numpy(), 'image_opticalflow': data['image_opticalflow']}

class NormalizeOpt(object):

    def __init__(self, mean, std):
        self.mean = list(map(float, mean))
        self.std = list(map(float, std))

    def __call__(self, data): # tensor                                                              
        # TODO: make efficient                                                                      
        img = data['image']
        tensor = torch.FloatTensor(data['image_opticalflow'])
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s).float()

        return {'image': img, 'image_opticalflow': tensor.numpy()}

    
normalize_opt = NormalizeOpt(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

# Normalization transform
normalize = NormalizeWithName(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])



def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize, normalize_opt])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()
    
    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size
        count_num = i
        image_img = image['image']
        image_opt = image['image_opticalflow']
        image_img = image_img.to(device)
        image_opt = image_opt.to(device)

        image_img = encoder(image_img)
        image_opt = encoder_opt(image_opt)
        

        # Encode
        encoder_out = image_img #encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim
        encoder_out_opt = image_opt
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        encoder_out_opt = encoder_out_opt.view(1, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
       
        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)# (k, num_pixels, encoder_dim)
        encoder_out_opt = encoder_out_opt.expand(k, num_pixels, encoder_dim)

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
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            awe_opt, _ = decoder.attention_opt(encoder_out_opt, h)

            
            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe_fuz = awe + awe_opt
            awe = gate * awe_fuz

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
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
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            encoder_out_opt = encoder_out_opt[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
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

  #      if count_num == 10:
   #         break


    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)
    print("references = {}".format(references))
    print("hypotheses = {}".format(hypotheses))
    # refs_dict: {id: [{id: sentence}, {id: sentence}, {id: sentence}, {id: sentence}, {id: sentence}], ...}
    # hypo_dict: {id: [{id: sentence}], ...}
    ref_dict  = {}
    hypo_dict = {}

    # Calculate BLEU, METEOR, ROUGE_L, CIDEr scores
    for idx_ref, refs in enumerate(references):
        # 5sentences list
        sent5_list = []

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
    with open('./results/ref_dict_Version00_checkpoint_msrvtt2016_5_cap.pickle', mode='wb') as f:
        pickle.dump(ref_dict, f, protocol=2)
    with open('./results/hypo_dict_Version00_checkpoint_msrvtt2016_5_cap.pickle', mode='wb') as g:
        pickle.dump(hypo_dict, g, protocol=2)

#['a', 'd', 'g']
#['b', 'e', 'h']
#['c', 'f', 'i']

        

#    print("hypo_dict = {}".format(hypo_dict))
#    sys.exit()
        
    # ref, dictionary: {id: sentence}
    # hypo, dictionary): {id: sentence}
#    scores = [
 #       (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
  #      (Meteor(), "METEOR"),
   #     (Rouge(), "ROUGE_L"),
    #    (Cider(), "CIDEr")
     #   ]
    #final_scores = {}

   # for scorer, method in scorers:
    #    score, scores = scorer.compute_score(ref_dict, hypo_dict)
     #   if type(score) == list:
      #      for m, s in        #         final_scores[m] = s

        #else:
         #   final_scores[method] = score
    
    #print(final_scores)            
    
    return bleu4


if __name__ == '__main__':
    beam_size = 3
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))

