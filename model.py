""" Your SkipGram Word2Vec model. """

import torch


class Model(torch.nn.Module):
    """SkipGram Word2Vec model."""

    def __init__(self, vocab_size):
        super(Model, self).__init__()
        self.embedding_dim = 64
        
        self.embeddings = torch.nn.Embedding(vocab_size, self.embedding_dim)
        torch.nn.init.uniform_(self.embeddings.weight, a=-0.1, b=0.1)
        
        self.linear = torch.nn.Linear(self.embedding_dim, vocab_size, bias=False)

        return

    def forward(self, input_ids):
        """Forward pass of the model.

        Args:
        input_ids: torch.Tensor, the input token indices [batch_size]

        Returns:
        logits: torch.Tensor, the output logits [batch_size, vocab_size]
        """
        embeddings = self.embeddings(input_ids)
        logits = self.linear(embeddings)
        return logits

    def get_embeddings(self):
        """Return the embedding matrix.
        Returns:
        torch.Tensor, the embedding [vocab_size, embedding_dim]
        """
        return self.embeddings.weight
