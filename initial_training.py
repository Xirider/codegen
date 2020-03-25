from tensor2tensor.utils.adafactor import AdafactorOptimizer
import gpt_2_simple as gpt2
from datetime import datetime

file_name = "text_encoded.npz"
# file_name = "python/final/jsonl/train/gptready.txt"
full_frame = None

full_frame = None

sess = gpt2.start_tf_sess()
# 774M
gpt2.finetune(sess,
              dataset=file_name,
              model_name='1558M',
              steps=-1,
              restore_from='fresh',
              run_name='run1',
              print_every=10,
              sample_every=250,
              save_every=500,
              use_memory_saving_gradients=True,
              only_train_transformer_layers=True,
              accumulate_gradients=1,
              optimizer="adafactor",
              fp16=False,
              output_checkpoint=True,
              batch_size=1
              
              )