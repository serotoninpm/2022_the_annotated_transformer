from data import *
from utils.greedy_decode import greedy_decode
from utils.make_model import make_model
from utils.bleu import get_bleu


model = make_model(len(loader.source.vocab), len(loader.target.vocab), N=n_layers, d_model=d_model, d_ff=ffn_hidden, h=n_heads, dropout=dropout)
model.to(device)
model.load_state_dict(torch.load("./saved/model-0.018100541085004807.pt"))


def test_model():
    bleu_list = []
    for i, batch in enumerate(test_iter):
        for batch_count in range(len(batch.src)):
            src = batch.src[batch_count:batch_count+1]
            src_mask = (src != loader.source.vocab.stoi["<blank>"]).unsqueeze(-2)
            trg = batch.trg[batch_count]
            out = greedy_decode(model, src, src_mask,
                                max_len=60, start_symbol=loader.target.vocab.stoi["<s>"])
            print("Translation:", end="\t")
            trans_result =''
            for word in range(1, out.size(1)):
                sym = loader.target.vocab.itos[out[0, word]]
                if sym == "</s>": break
                trans_result += sym + " "
            trans_result = trans_result.strip()
            print(trans_result)
            print("Target:", end="\t")
            ground_truth = ""
            for word in range(1, trg.size(0)):
                sym = loader.target.vocab.itos[trg.data[word]]
                if sym == "</s>": break
                ground_truth += sym + " "
            ground_truth = ground_truth.strip()
            print(ground_truth)
            bleu = get_bleu(hypotheses=trans_result.split(), reference=ground_truth.split())
            print(bleu)
            print()
            bleu_list.append(bleu)
    print('TOTAL BLEU SCORE = {}'.format(sum(bleu_list)/len(bleu_list)))


if __name__ == '__main__':
    test_model()
