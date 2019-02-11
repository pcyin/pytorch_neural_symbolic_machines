import json
import numpy as np

import torch
import torch.nn as nn


class Embedder(nn.Module):
    def __init__(self, trainable_token_num: int, embed_size: int, pretrained_embedding: torch.Tensor):
        super(Embedder, self).__init__()
        
        self.embed_size = embed_size
        self.pretrained_embed_linear = nn.Linear(pretrained_embedding.size(-1), embed_size)

        self.pretrained_embedding = nn.Embedding(pretrained_embedding.size(0), pretrained_embedding.size(1),
                                                 _weight=pretrained_embedding)
        self.pretrained_embedding.weight.requires_grad = False
        self.trainable_token_num = trainable_token_num
        self.trainable_embedding = nn.Embedding(trainable_token_num, embed_size)

    @property
    def device(self):
        return self.pretrained_embedding.weight.device

    def forward(self, x):
        is_pretrained_token = torch.ge(x, self.trainable_token_num).float()
        pretrained_token_ids = torch.max(x - self.trainable_token_num, torch.zeros_like(x, device=x.device)).long()
        pretrained_embeddings = self.pretrained_embedding(pretrained_token_ids) * is_pretrained_token.unsqueeze(-1)
        pretrained_embeddings = self.pretrained_embed_linear(pretrained_embeddings)

        is_trainable_token = 1 - is_pretrained_token.float()
        trainable_token_ids = x * is_trainable_token.long()
        trainable_embeddings = self.trainable_embedding(trainable_token_ids) * is_trainable_token.unsqueeze(-1)

        embeddings = trainable_embeddings + pretrained_embeddings

        return embeddings


class EmbeddingModel(object):
    def __init__(
            self, vocab_file, embedding_file, normalize_embeddings=True):
        with open(embedding_file, 'rb') as f:
            self.embedding_mat = np.load(f)
        if normalize_embeddings:
            self.embedding_mat = self.embedding_mat / np.linalg.norm(
                self.embedding_mat, axis=1, keepdims=True)
        with open(vocab_file, 'r') as f:
            tks = json.load(f)
        self.vocab = dict(zip(tks, range(len(tks))))

    def __contains__(self, word):
        return word in self.vocab

    def __getitem__(self, word):
        if word in self.vocab:
            index = self.vocab[word]
            return self.embedding_mat[index]
        else:
            raise KeyError
