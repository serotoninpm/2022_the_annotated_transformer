import math
import time
from data import *
from utils.make_model import make_model
from utils.label_smoothing import LabelSmoothing
from utils.NoamOpt import NoamOpt
from utils.run_epoch import run_epoch
from utils.simple_loss_compute import SimpleLossCompute
from utils.epoch_time import epoch_time
from utils.batch import rebatch


pad_idx = loader.target.vocab.stoi['<blank>']
model = make_model(len(loader.source.vocab), len(loader.target.vocab), N=n_layers, d_model=d_model, d_ff=ffn_hidden, h=n_heads, dropout=dropout)
model.to(device)
criterion = LabelSmoothing(size=len(loader.target.vocab), padding_idx=pad_idx, smoothing=0.1)
criterion.to(device)
model_opt = NoamOpt(model.src_embed[0].d_model, factor, warmup,
          torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def run(total_epoch, best_loss):
    train_losses, valid_losses = [], []
    for step in range(total_epoch):
        start_time = time.time()
        # train epoch
        model.train()
        train_loss = run_epoch((rebatch(pad_idx, b) for b in train_iter), model, SimpleLossCompute(model.generator,
                                                                                                   criterion, model_opt))

        # valid epoch
        model.eval()
        valid_loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model, SimpleLossCompute(model.generator,
                                                                                                   criterion, None))
        end_time = time.time()

        train_losses.append(train_loss.item())
        valid_losses.append(valid_loss.item())
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/valid_loss.txt', 'w')
        f.write(str(valid_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
