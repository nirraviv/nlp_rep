import os, sys

import torch


def set_work_dir(local_path="ner_bert_crf", server_path="ner_bert_crf"):
    if os.path.exists(os.getenv("HOME")+'/'+local_path):
        os.chdir(os.getenv("HOME")+'/'+local_path)
    elif os.path.exists(os.getenv("HOME")+'/'+server_path):
        os.chdir(os.getenv("HOME")+'/'+server_path)
    else:
        raise Exception('Set work path error!')


def get_data_dir(local_path="ner_bert_crf", server_path="ner_bert_crf"):
    if os.path.exists(os.getenv("HOME")+'/'+local_path):
        return os.getenv("HOME")+'/'+local_path
    elif os.path.exists(os.getenv("HOME")+'/'+server_path):
        return os.getenv("HOME")+'/'+server_path
    else:
        raise Exception('get data path error!')


print('Python version ', sys.version)
print('PyTorch version ', torch.__version__)

# set_work_dir()
print('Current dir:', os.getcwd())

cuda_yes = torch.cuda.is_available()
# cuda_yes = False
print('Cuda is available?', cuda_yes)
device = torch.device("cuda:0" if cuda_yes else "cpu")
print('Device:', device)

data_dir = r'C:\project\NeuroNER\neuroner\data\conll2003\en'
# "Whether to run training."
do_train = True
# "Whether to run eval on the dev set."
do_eval = True
# "Whether to run the model in inference mode on the test set."
do_predict = True
# Whether load checkpoint file before train model
load_checkpoint = True
# "The vocabulary file that the BERT model was trained on."
max_seq_length = 180 #256
batch_size = 32 #32
# "The initial learning rate for Adam."
learning_rate0 = 3e-5
lr0_crf_fc = 3e-5
weight_decay_finetune = 1e-5 #0.01
weight_decay_crf_fc = 5e-6 #0.005
total_train_epochs = 15
gradient_accumulation_steps = 1
warmup_proportion = 0.1
output_dir = './output/'
bert_model_scale = 'bert-base-cased'
do_lower_case = False
# eval_batch_size = 8
# predict_batch_size = 8
# "Proportion of training to perform linear learning rate warmup for. "
# "E.g., 0.1 = 10% of training."
# warmup_proportion = 0.1
# "How often to save the model checkpoint."
# save_checkpoints_steps = 1000
# "How many steps to make in each estimator call."
# iterations_per_loop = 1000
