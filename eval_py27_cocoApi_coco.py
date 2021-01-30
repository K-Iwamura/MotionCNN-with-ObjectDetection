from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

import cPickle


# load ref and hypo dictionary
# default
with open('./results/ref_dict_Version00_checkpoint_coco_5_cap.pickle', mode='rb') as fi:
    ref_dict = cPickle.load(fi)

# hypo
# default
with open('./results/hypo_dict_Version00_checkpoint_coco_5_cap.pickle', mode='rb') as fg:
    hypo_dict = cPickle.load(fg)



#tokenizer = PTBTokenizer()
#ref = tokenizer.tokenize(ref)
#hypo = tokenizer.tokenize(hypo)

#######print "ref = {}".format(ref_dict)
#######print "hypo = {}".format(hypo_dict)
# culculate scores
scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Meteor(), "METEOR"),
    (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr")
    ]

final_scores = {}
for scorer, method in scorers:
    score, scores = scorer.compute_score(ref_dict, hypo_dict)
    if type(score) == list:
        for m, s in zip(method, score):
            final_scores[m] = s
    else:
        final_scores[method] = score

print "final_scores = {}".format(final_scores)
