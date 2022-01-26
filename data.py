from config import *
from utils.tokenizer import Tokenizer
from utils.data_loader import DataLoader


tokenizer = Tokenizer()
loader = DataLoader(ext=('.en', '.de'),
                    tokenize_en=tokenizer.tokenize_en,
                    tokenize_de=tokenizer.tokenize_de,
                    init_token='<s>',
                    eos_token='</s>',
                    pad_token='<blank>')

# 데이터셋 만들기
train, valid, test = loader.make_dataset()

# vocab 생성
loader.build_vocab(train_data=train, min_freq=2)

# iterable 하게 만들기
train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                                     batch_size=BATCH_SIZE,
                                                     device=device)
