module Jax.NN.Embed
  ( embed
  , unembed
  ) where

import Prelude

import Effect (Effect)
import Jax.Core (D1, D2, NDArray, matmul, ref, take, transpose)

-- | Token embedding lookup: `table[ids]`.
-- |
-- | * `table` — `[vocab_size, embed_dim]` (D2). Typically `LongLived`.
-- | * `ids`   — `[seq_len]` (D1, int32 token IDs).
-- | * result  — `[seq_len, embed_dim]` (D2), refcount 1.
-- |
-- | Both inputs are borrowed; their refcounts are unchanged on return.
embed
  :: NDArray D2
  -> NDArray D1
  -> Effect (NDArray D2)
embed table ids = do
  tableR <- ref table
  idsR <- ref ids
  take tableR idsR 0

-- | Language-model head: project hidden states to vocabulary logits via
-- | the (transposed) embedding table — i.e. weight-tied with `embed`.
-- |
-- | * `hidden` — `[seq_len, embed_dim]` (D2).
-- | * `table`  — `[vocab_size, embed_dim]` (D2). Same table fed to
-- |   `embed`; we transpose internally.
-- | * result   — `[seq_len, vocab_size]` (D2).
-- |
-- | Inputs borrowed.
unembed
  :: NDArray D2
  -> NDArray D2
  -> Effect (NDArray D2)
unembed hidden table = do
  tableR <- ref table
  tableT <- transpose tableR
  hiddenR <- ref hidden
  matmul hiddenR tableT
