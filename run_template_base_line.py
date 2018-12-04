import torch
from loader import LoaderHandler
from evaluator import Merics
from trainer import Trainer, LangTrainer
from model import Seq2seq, Criterion, languageModel
from utils import ConfigParser, utils
import fileinput

def runVal(config):
    pos_f = open('test_pos_result')
    pos_dat = pos_f.readlines()
    pos_f.close()
    neg_f = open('test_neg_result')
    neg_dat = neg_f.readlines()
    neg_f.close()
    pos_dt = []
    for i in pos_dat:
        if i[0] == 't':
            new = [i[7:]]
        elif i[0] == 'p':
            new.append(i[6:])
        else:
            pos_dt.append(new)
    pos_dt = sorted(pos_dt, key = lambda x:x[0])
    neg_dt = []
    for i in neg_dat:
        if i[0] == 't':
            new = [i[7:]]
        elif i[0] == 'p':
            new.append(i[6:])
        else:
            neg_dt.append(new)
    neg_dt = sorted(neg_dt, key = lambda x:x[0])
    pred_pos = [i[1].split() for i in pos_dt]
    pred_neg = [i[1].split() for i in neg_dt]
    preds = {'positive': pred_pos, 'negative': pred_neg}
    with open("../AuxData/wordDict_classifier","rb") as fp:
        word_to_id = pickle.load(fp)
    attention_model = StructuredSelfAttention_test(batch_size=1,lstm_hid_dim=100,d_a = 100,r=2,vocab_size=len(word_to_id),max_len=25,type=0,n_classes=1,use_pretrained_embeddings=False,embeddings=None)		
    evaluateMetrics = Metrics(config["metric"]["classifier_weight_path"], config["metric"]["ref_file"], attention_model,"../AuxData/wordDict_classifier" ,config)
    


def main():
	config = ConfigParser.parse_config()
	mode = config['opt'].mode
	if mode == 'train':
		runTrain(config)
	elif mode =='pretrain':
		runPreTrain(config)
	elif mode == 'val':
		runVal(config)
	elif mode == 'test':
		runVal(config)
	elif mode == 'online':
		runOnline(config)
	else:
		pass
	
if __name__ == '__main__':
	main() 
