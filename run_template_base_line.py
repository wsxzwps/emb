import torch
from loader import LoaderHandler
from evaluator import Merics
from trainer import Trainer, LangTrainer
from model import Seq2seq, Criterion, languageModel
from utils import ConfigParser, utils
import fileinput

def runVal(config):
    ####
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
