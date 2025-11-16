import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu , SmoothingFunction


def sentence_to_words(sentence):
    words = sentence.split()
    last = words.pop()
    last, end = last[:-1], last[-1] # seperate the punctuation
    words.append(last)
    words.append(end)
    return words

make_word_onehot = lambda word, vocab: nn.functional.one_hot(torch.tensor(vocab.get_index(word)), num_classes=vocab.vocab_size).to(torch.float32)


class TrEngDataset(Dataset): # data to be downloaded from https://www.manythings.org/anki/
    def __init__(self, path=None, max_sentence_length=None):
        self.path = "data/tur-eng/tur.txt" if not path else path 
        self.data = []
        self._wordlimits = [84, 1123, 28555]
        count = 0
        with open(self.path, "r") as f:
            line = f.readline()
            while line:
                count+=1
                if max_sentence_length and self._wordlimits[max_sentence_length-1] < count:
                    break
                eng, tr = line.split('\t')[:2]
                self.data.append((tr, eng))
                line = f.readline()


    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self):
        return len(self.data)


class Vocabulary():
    def __init__(self, sentence_list):
        self.vocab = {}
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.pad_token = "<PAD>"
        self.vocab_size = 0
        for token in [self.sos_token, self.eos_token, self.pad_token]:
            self.vocab[token] = self.vocab_size
            self.vocab_size+=1

        for sentence in sentence_list:
            words = sentence_to_words(sentence)
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = self.vocab_size
                    self.vocab_size+=1

        self.i2w = {index: word for word, index in self.vocab.items()}
    
    def get_word(self, index):
        return self.i2w[index]
    
    def get_index(self, word):
        return self.vocab.get(word, 'NAN')


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        combined_size = input_size + hidden_size
        # forget gate
        self.forget = nn.Linear(combined_size, hidden_size, bias=True)
        # update gate
        self.update = nn.Linear(combined_size, hidden_size, bias=True)
        # candidate gate
        self.candidate = nn.Linear(combined_size, hidden_size, bias=True)
        # output gate
        self.output = nn.Linear(combined_size, hidden_size, bias=True)
    
    def forward(self, x, prev_a, prev_c):
        input_vector = torch.concat((x, prev_a))
        forget = torch.sigmoid(self.forget(input_vector))
        update = torch.sigmoid(self.update(input_vector))
        candidate = torch.tanh(self.candidate(input_vector))
        output = torch.sigmoid(self.output(input_vector))

        c = (forget * prev_c) + (update * candidate)
        a =  torch.tanh(c) * output

        return a, c

class LSTMStacked(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, output_size):
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.Sequential(LSTMCell(input_size, hidden_size), *(LSTMCell(hidden_size, hidden_size) for _ in range(self.num_layers)))
        self.final = nn.Linear(hidden_size, output_size, bias=True)
    
    def forward(self, x, prev_a_c_list):
        new_a_c_list = []
        for c in range(self.num_layers):
            prev_a, prev_c = prev_a_c_list[c]
            new_a, new_c = self.cells[c](x, prev_a, prev_c)
            new_a_c_list.append((new_a, new_c))
            x = new_a
        
        x = self.final(x)
        return x, new_a_c_list


class Seq2Seq(nn.Module):
    def __init__(self, vocab, output_vocab, hidden_size, num_layers=3, sos_token='<SOS>'):
        super().__init__()
        self.vocab = vocab
        self.output_vocab = output_vocab
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sos_token = sos_token
        self.eos_token_indexes = [self.output_vocab.get_index(token) for token in [self.vocab.eos_token, '.', '!', '?']]
        self.encoder = LSTMStacked(self.num_layers, self.vocab.vocab_size, hidden_size, hidden_size)
        self.decoder = LSTMStacked(self.num_layers, self.output_vocab.vocab_size, hidden_size, self.output_vocab.vocab_size)
    
    def forward(self, seq, target_seq):
        a_c_0 = [(torch.zeros(self.hidden_size), torch.zeros(self.hidden_size)) for _ in range(self.num_layers)]
        prev_a_c = a_c_0
        for word in seq:
            _, prev_a_c = self.encoder(word, prev_a_c)

        input = make_word_onehot(self.sos_token, self.output_vocab)
        target_seq = [input] + target_seq
        output = []
        for target_word in target_seq:
            input, prev_a_c = self.decoder(target_word, prev_a_c)
            output.append(input)
        
        return output

    def predict(self, seq):
        a_c_0 = [(torch.zeros(self.hidden_size), torch.zeros(self.hidden_size)) for _ in range(self.num_layers)]
        prev_a_c = a_c_0
        for word in seq:
            _, prev_a_c = self.encoder(word, prev_a_c)

        input = make_word_onehot(self.sos_token, self.output_vocab)
        output = []
        
        while torch.argmax(input) not in self.eos_token_indexes and len(output) < len(seq)+5: 
            input, prev_a_c = self.decoder(input, prev_a_c)
            output.append(input)
        
        return output


def pad_couple(first, sec, pad):
    while len(first) > len(sec):
        sec.append(pad)
    while len(sec) > len(first):
        first.append(pad)
    return first, sec


dataset = TrEngDataset(max_sentence_length=2) #limit the number of words from dataset
# build vocab
turkish_sentences = np.array(dataset.data)[:,0]
tr_vocab = Vocabulary(turkish_sentences)
english_sentences = np.array(dataset.data)[:,1]
eng_vocab = Vocabulary(english_sentences)


def train():
    torch.set_grad_enabled(True)

    model = Seq2Seq(tr_vocab, eng_vocab, hidden_size=64)
    initial_lr = 1e-3
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    EPOCHS = 1

    for ep in range(EPOCHS):
        epoch_loss = 0
        count = 0
        for s in range(len(turkish_sentences)):
            count+=1
            # print(f"Going for the {count}. with input {turkish_sentences[s]} and output {english_sentences[s]}")
            optimizer.zero_grad()
            
            sentence = turkish_sentences[s]
            word_list = sentence_to_words(sentence)
            embedding_list = [make_word_onehot(word, tr_vocab) for word in word_list]

            target_sentence = english_sentences[s]
            target_list = sentence_to_words(target_sentence)
            target_embedding_list = [make_word_onehot(word, eng_vocab) for word in target_list]

            logits = model(embedding_list, target_embedding_list)

            if len(logits) != len(target_embedding_list):
                logits, target_embedding_list = pad_couple(logits, target_embedding_list, make_word_onehot(eng_vocab.pad_token, eng_vocab))
            
            loss = 0
            for i in range(len(logits)):
                loss += criterion(logits[i], target_embedding_list[i])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print("EPOCH", ep, epoch_loss / len(turkish_sentences))
        lr_scheduler.step(epoch_loss / len(turkish_sentences))
    return model


def test(model):
    torch.set_grad_enabled(False)
    try:
        chencherry = SmoothingFunction()
        test_count = 0
        outputs = []
        targets = []
        total_bleu_score = 0
        for s in range(len(turkish_sentences)):
            test_count+=1
            print(f"Going for the {test_count}. with input {turkish_sentences[s]} and output {english_sentences[s]}")
            
            sentence = turkish_sentences[s]
            word_list = sentence_to_words(sentence)
            embedding_list = [make_word_onehot(word, tr_vocab) for word in word_list]

            target_sentence = english_sentences[s]
            target_list = sentence_to_words(target_sentence)
            target_embedding_list = [make_word_onehot(word, eng_vocab) for word in target_list]

            logits = model.predict(embedding_list)
            probs = [torch.softmax(logit, 0) for logit in logits]
            output = [eng_vocab.get_word(torch.argmax(prob).item()) for prob in probs]
            while output[-1] == eng_vocab.pad_token:
                output.pop()
            outputs.append(output)
            targets.append([target_list])

            print('Output', output)
            print('Target', target_list)
            score = sentence_bleu([target_list], output, smoothing_function=chencherry.method2)
            total_bleu_score += score
            print('Score', score)

        print('Total Avg Score', total_bleu_score / len(turkish_sentences))
        
    finally:
        torch.set_grad_enabled(True)

def main():
    model = train()
    test(model)
    # torch.save(model.state_dict, 'models/seq2seq_2word_ep1.pth')

if __name__=='__main__':
    main()