import os
from typing import List, Tuple
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "./data"
RESULT_DIR = "./results"

def read_conll_file(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    sentences = []
    tags = []
    cur_tokens = []
    cur_tags = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur_tokens:
                    sentences.append(cur_tokens)
                    tags.append(cur_tags)
                    cur_tokens, cur_tags = [], []
                continue
            cols = line.split()
            token = cols[0]
            tag = cols[-1]
            cur_tokens.append(token)
            cur_tags.append(tag)

    if cur_tokens:
        sentences.append(cur_tokens)
        tags.append(cur_tags)
    return sentences, tags


def build_vocab(sentences: List[List[str]], min_freq: int = 1):
    counter = Counter()
    for sent in sentences:
        counter.update(sent)

    word2id = {"<PAD>": 0, "<UNK>": 1}
    for w, c in counter.items():
        if c >= min_freq:
            word2id[w] = len(word2id)
    id2word = {i: w for w, i in word2id.items()}
    return word2id, id2word


def build_tag_vocab(tags: List[List[str]]):
    tag2id = {}
    for sent_tags in tags:
        for t in sent_tags:
            if t not in tag2id:
                tag2id[t] = len(tag2id)
    id2tag = {i: t for t, i in tag2id.items()}
    return tag2id, id2tag


def build_char_vocab(sentences: List[List[str]]):
    char2id = {"<PAD>": 0, "<UNK>": 1}
    for sent in sentences:
        for w in sent:
            for ch in w:
                if ch not in char2id:
                    char2id[ch] = len(char2id)
    id2char = {i: c for c, i in char2id.items()}
    return char2id, id2char


class NERDataset(Dataset):
    def __init__(self, sentences, tags, word2id, tag2id, char2id=None):
        self.sentences = sentences
        self.tags = tags
        self.word2id = word2id
        self.tag2id = tag2id
        self.char2id = char2id

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        tags = self.tags[idx]

        word_ids = [self.word2id.get(w, self.word2id["<UNK>"]) for w in words]
        tag_ids = [self.tag2id[t] for t in tags]

        if self.char2id is not None:
            char_ids = []
            for w in words:
                chars = [self.char2id.get(ch, self.char2id["<UNK>"]) for ch in w]
                char_ids.append(chars)
        else:
            char_ids = None

        return (
            torch.tensor(word_ids, dtype=torch.long),
            torch.tensor(tag_ids, dtype=torch.long),
            char_ids,
        )


def collate_fn(batch):
    batch_word_ids, batch_tag_ids, batch_char_ids = zip(*batch)
    lengths = torch.tensor([len(x) for x in batch_word_ids], dtype=torch.long)

    max_len = lengths.max().item()
    pad_word_id = 0
    pad_tag_id = -1

    padded_words, padded_tags = [], []

    use_char = batch_char_ids[0] is not None
    if use_char:
        max_char_len = 1
        for sent_chars in batch_char_ids:
            for w_chars in sent_chars:
                if len(w_chars) > max_char_len:
                    max_char_len = len(w_chars)
        padded_chars = torch.zeros(len(batch_word_ids), max_len, max_char_len, dtype=torch.long)
    else:
        padded_chars = None

    for i, (w_ids, t_ids) in enumerate(zip(batch_word_ids, batch_tag_ids)):
        pad_len = max_len - len(w_ids)
        padded_words.append(torch.cat([w_ids, torch.full((pad_len,), pad_word_id, dtype=torch.long)]))
        padded_tags.append(torch.cat([t_ids, torch.full((pad_len,), pad_tag_id, dtype=torch.long)]))

        if use_char:
            sent_chars = batch_char_ids[i]
            for j, w_chars in enumerate(sent_chars):
                padded_chars[i, j, :len(w_chars)] = torch.tensor(w_chars, dtype=torch.long)

    padded_words = torch.stack(padded_words, dim=0)
    padded_tags = torch.stack(padded_tags, dim=0)
    return padded_words, padded_tags, lengths, padded_chars


"""
>>> Models
"""

class BiLSTMTagger(nn.Module):
    # BiLSTM + softmax model
    def __init__(self, vocab_size, tagset_size,
                 embedding_dim=100, hidden_dim=256,
                 pad_idx=0, char_vocab_size=None,
                 char_emb_dim=30, char_pad_idx=0):
        super().__init__()
        self.use_char = char_vocab_size is not None
        self.pad_idx = pad_idx
        self.char_pad_idx = char_pad_idx

        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        if self.use_char:
            self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=char_pad_idx)
            lstm_input_dim = embedding_dim + char_emb_dim
        else:
            lstm_input_dim = embedding_dim

        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def _combine_embeddings(self, sentences, chars=None):
        word_emb = self.word_embedding(sentences)  # (B,T,Ew)
        if self.use_char and chars is not None:
            char_emb = self.char_embedding(chars)   # (B,T,C,Ec)
            mask = (chars != self.char_pad_idx).unsqueeze(-1)
            char_sum = (char_emb * mask).sum(2)
            char_len = mask.sum(2).clamp(min=1)
            char_rep = char_sum / char_len         # (B,T,Ec)
            return torch.cat([word_emb, char_rep], dim=-1)
        else:
            return word_emb

    def forward(self, sentences, lengths, chars=None):
        embeds = self._combine_embeddings(sentences, chars)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        emissions = self.hidden2tag(lstm_out)      # (B,T,C)
        return emissions


class BiLSTM_CRF(nn.Module):
    # BiLSTM + CRF
    def __init__(self, vocab_size, tagset_size,
                 embedding_dim=100, hidden_dim=256,
                 pad_idx=0, char_vocab_size=None,
                 char_emb_dim=30, char_pad_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.pad_idx = pad_idx
        self.use_char = char_vocab_size is not None
        self.char_pad_idx = char_pad_idx

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        if self.use_char:
            self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=char_pad_idx)
            lstm_input_dim = embedding_dim + char_emb_dim
        else:
            lstm_input_dim = embedding_dim

        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.start_transitions = nn.Parameter(torch.empty(tagset_size))
        self.end_transitions = nn.Parameter(torch.empty(tagset_size))
        self.transitions = nn.Parameter(torch.empty(tagset_size, tagset_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        if self.use_char:
            nn.init.uniform_(self.char_embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.hidden2tag.weight)
        nn.init.constant_(self.hidden2tag.bias, 0.0)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _combine_embeddings(self, sentences, chars=None):
        word_emb = self.embedding(sentences)
        if self.use_char and chars is not None:
            char_emb = self.char_embedding(chars)
            mask = (chars != self.char_pad_idx).unsqueeze(-1)
            char_sum = (char_emb * mask).sum(2)
            char_len = mask.sum(2).clamp(min=1)
            char_rep = char_sum / char_len
            return torch.cat([word_emb, char_rep], dim=-1)
        else:
            return word_emb

    def _compute_emissions(self, sentences, lengths, chars=None):
        embeds = self._combine_embeddings(sentences, chars)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    @staticmethod
    def _log_sum_exp(tensor, dim=-1):
        max_score, _ = tensor.max(dim)
        max_score_broadcast = max_score.unsqueeze(dim)
        return max_score + torch.log(torch.sum(torch.exp(tensor - max_score_broadcast), dim))

    def _compute_log_partition(self, emissions, mask):
        B, T, C = emissions.size()
        log_prob = self.start_transitions + emissions[:, 0]

        for t in range(1, T):
            emit_t = emissions[:, t]
            mask_t = mask[:, t].unsqueeze(1)
            score_t = log_prob.unsqueeze(2) + self.transitions + emit_t.unsqueeze(1)
            log_prob_t = self._log_sum_exp(score_t, dim=1)
            log_prob = torch.where(mask_t, log_prob_t, log_prob)

        log_prob = log_prob + self.end_transitions
        return self._log_sum_exp(log_prob, dim=1)

    def _compute_gold_score(self, emissions, tags, mask):
        B, T, C = emissions.size()
        score = self.start_transitions[tags[:, 0]]
        score += emissions[torch.arange(B), 0, tags[:, 0]]

        for t in range(1, T):
            mask_t = mask[:, t]
            emit_t = emissions[torch.arange(B), t, tags[:, t]]
            trans_t = self.transitions[tags[:, t - 1], tags[:, t]]
            score += (emit_t + trans_t) * mask_t

        last_tag_indices = mask.long().sum(1) - 1
        last_tags = tags[torch.arange(B), last_tag_indices]
        score += self.end_transitions[last_tags]
        return score

    def neg_log_likelihood(self, sentences, tags, lengths, chars=None):
        max_len = sentences.size(1)
        mask = (torch.arange(max_len, device=sentences.device)
                .unsqueeze(0) < lengths.unsqueeze(1))
        tags = tags.clone()
        tags[~mask] = 0

        emissions = self._compute_emissions(sentences, lengths, chars)
        log_partition = self._compute_log_partition(emissions, mask)
        gold_score = self._compute_gold_score(emissions, tags, mask)
        return (log_partition - gold_score).mean()

    def _viterbi_decode(self, emissions, mask):
        B, T, C = emissions.size()
        score = self.start_transitions + emissions[:, 0]
        history = []

        for t in range(1, T):
            emit_t = emissions[:, t]
            mask_t = mask[:, t].unsqueeze(1)
            score_t = score.unsqueeze(2) + self.transitions
            best_score, best_path = score_t.max(1)
            best_score = best_score + emit_t
            score = torch.where(mask_t, best_score, score)
            history.append(best_path)

        score = score + self.end_transitions
        best_final_score, best_last_tag = score.max(1)

        best_paths = []
        for i in range(B):
            seq_len = mask[i].sum().item()
            last_tag = best_last_tag[i].item()
            path = [last_tag]
            for hist in reversed(history[:seq_len - 1]):
                last_tag = hist[i][last_tag].item()
                path.append(last_tag)
            path.reverse()
            best_paths.append(path)
        return best_paths

    def forward(self, sentences, lengths, chars=None):
        max_len = sentences.size(1)
        mask = (torch.arange(max_len, device=sentences.device)
                .unsqueeze(0) < lengths.unsqueeze(1))
        emissions = self._compute_emissions(sentences, lengths, chars)
        return self._viterbi_decode(emissions, mask)


def evaluate(model, dataloader, model_type="crf"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            sentences, tags, lengths, chars = batch
            sentences = sentences.to(DEVICE)
            tags = tags.to(DEVICE)
            lengths = lengths.to(DEVICE)
            if chars is not None:
                chars = chars.to(DEVICE)

            if model_type == "crf":
                best_paths = model(sentences, lengths, chars)
            else:
                emissions = model(sentences, lengths, chars)
                pred_ids = emissions.argmax(-1)
                best_paths = []
                for i in range(sentences.size(0)):
                    L = lengths[i].item()
                    best_paths.append(pred_ids[i, :L].tolist())

            for i, path in enumerate(best_paths):
                gold = tags[i][:lengths[i]].tolist()
                pred = path
                assert len(gold) == len(pred)
                for g, p in zip(gold, pred):
                    if g == p:
                        correct += 1
                    total += 1

    acc = correct / total if total > 0 else 0.0
    return acc


def encode_sentence(sentence: str, word2id, char2id=None):
    words = sentence.split()
    word_ids = [word2id.get(w, word2id["<UNK>"]) for w in words]
    word_tensor = torch.tensor([word_ids], dtype=torch.long)
    lengths = torch.tensor([len(words)], dtype=torch.long)

    if char2id is None:
        return words, word_tensor, lengths, None

    max_char_len = max(len(w) for w in words)
    char_tensor = torch.zeros(1, len(words), max_char_len, dtype=torch.long)
    for i, w in enumerate(words):
        for j, ch in enumerate(w):
            char_tensor[0, i, j] = char2id.get(ch, char2id["<UNK>"])

    return words, word_tensor, lengths, char_tensor

def predict_sentence(model, sentence: str,
                     word2id, id2tag,
                     model_type="crf", char2id=None):
    """
    Return: list of (word, tag_str)
    """
    model.eval()
    words, word_tensor, lengths, char_tensor = encode_sentence(sentence, word2id, char2id)
    word_tensor = word_tensor.to(DEVICE)
    lengths = lengths.to(DEVICE)
    if char_tensor is not None:
        char_tensor = char_tensor.to(DEVICE)

    with torch.no_grad():
        if model_type == "crf":
            best_paths = model(word_tensor, lengths, char_tensor)
            tag_ids = best_paths[0]
        else:
            emissions = model(word_tensor, lengths, char_tensor)
            pred_ids = emissions.argmax(-1)
            L = lengths[0].item()
            tag_ids = pred_ids[0, :L].tolist()

    tags = [id2tag[int(i)] for i in tag_ids]
    return list(zip(words, tags))


def main():
    # read data and rebuild vocab
    train_sents, train_tags = read_conll_file(os.path.join(DATA_DIR, "train.txt"))
    dev_sents, dev_tags = read_conll_file(os.path.join(DATA_DIR, "dev.txt"))
    test_sents, test_tags = read_conll_file(os.path.join(DATA_DIR, "test.txt"))

    word2id, id2word = build_vocab(train_sents, min_freq=1)
    tag2id, id2tag = build_tag_vocab(train_tags)
    char2id, id2char = build_char_vocab(train_sents)

    print("Vocab size:", len(word2id))
    print("Tag size:", len(tag2id))
    print("Char vocab size:", len(char2id))

    # build datasets & dataloaders
    def make_loader(use_char: bool, split: str):
        sents = {"train": train_sents, "dev": dev_sents, "test": test_sents}[split]
        tgs = {"train": train_tags, "dev": dev_tags, "test": test_tags}[split]
        ds = NERDataset(
            sents,
            tgs,
            word2id,
            tag2id,
            char2id if use_char else None,
        )
        return DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    test_loader_word_only = make_loader(use_char=False, split="test")
    test_loader_word_char = make_loader(use_char=True, split="test")

    # load softmax model
    softmax_ckpt_path = os.path.join(RESULT_DIR, "bilstm_softmax_word_best.pt")
    softmax_model = BiLSTMTagger(
        vocab_size=len(word2id),
        tagset_size=len(tag2id),
        embedding_dim=100,
        hidden_dim=256,
        pad_idx=word2id["<PAD>"],
        char_vocab_size=None,  # word only
    ).to(DEVICE)
    softmax_state = torch.load(softmax_ckpt_path, map_location=DEVICE)["model_state_dict"]
    softmax_model.load_state_dict(softmax_state)

    # load CRF model
    crf_ckpt_path = os.path.join(RESULT_DIR, "bilstm_crf_word_char_best.pt")
    crf_model = BiLSTM_CRF(
        vocab_size=len(word2id),
        tagset_size=len(tag2id),
        embedding_dim=100,
        hidden_dim=256,
        pad_idx=word2id["<PAD>"],
        char_vocab_size=len(char2id),
        char_emb_dim=30,
        char_pad_idx=char2id["<PAD>"],
    ).to(DEVICE)
    crf_state = torch.load(crf_ckpt_path, map_location=DEVICE)["model_state_dict"]
    crf_model.load_state_dict(crf_state)

    # evaluate on test set
    softmax_test_acc = evaluate(softmax_model, test_loader_word_only, model_type="softmax")
    crf_test_acc = evaluate(crf_model, test_loader_word_char, model_type="crf")

    print("\n==== Test Accuracy ====")
    print(f"BiLSTM + Softmax (word only): {softmax_test_acc:.4f}")
    print(f"BiLSTM + CRF (word + char):   {crf_test_acc:.4f}")
    print("========================\n")

    # predict one sentence with both models
    demo_sentence = "Barack Obama visited Berlin yesterday ."
    print("Demo sentence:", demo_sentence)

    print("\n[Softmax model prediction]")
    pred_softmax = predict_sentence(
        softmax_model, demo_sentence,
        word2id=word2id,
        id2tag=id2tag,
        model_type="softmax",
        char2id=None,
    )
    for w, t in pred_softmax:
        print(f"{w:<15} {t}")


    print("\n[CRF model prediction]")
    pred_crf = predict_sentence(
        crf_model, demo_sentence,
        word2id=word2id,
        id2tag=id2tag,
        model_type="crf",
        char2id=char2id,
    )
    for w, t in pred_crf:
        print(f"{w:<15} {t}")


if __name__ == "__main__":
    main()
