-- | Character-level tokenizer.
-- |
-- | The smallest possible "tokenizer": one token per Unicode character
-- | observed in a corpus. The vocabulary is built from the unique
-- | characters of the input text, stably ordered by first occurrence
-- | so that two callers building from the same corpus produce the
-- | same encoding.
-- |
-- | Use case: educational/demo models like Karpathy's microGPT, where
-- | the dataset is plain text (e.g. names.txt, tiny shakespeare) and
-- | the goal is to keep the tokenizer transparent rather than ship
-- | with a real BPE vocabulary.
-- |
-- | Mirrors the Python pattern:
-- |
-- |     uchars = sorted(set(text))      # vocab
-- |     ctoi = {c: i for i, c in enumerate(uchars)}
-- |     encode = lambda s: [ctoi[c] for c in s]
-- |     decode = lambda ids: "".join(uchars[i] for i in ids)
-- |
-- | Build cost is O(N) over the input length. encode/decode are O(M)
-- | over the message length (Map lookup is O(log V) but V is small).
module Jax.Loaders.CharTokenizer
  ( CharTokenizer
  , fromText
  , encode
  , decode
  , size
  , chars
  ) where

import Data.Array as Array
import Data.Map (Map)
import Data.Map as Map
import Data.Maybe (Maybe(..))
import Data.String.CodeUnits (toCharArray, fromCharArray)
import Data.Tuple (Tuple(..))

newtype CharTokenizer = CharTokenizer
  { vocab :: Array Char       -- index -> char
  , ctoi  :: Map Char Int     -- char -> index
  }

-- | Build a tokenizer from a corpus. The vocabulary is the deduplicated
-- | set of characters in `text`, ordered by their first occurrence.
-- |
-- | First-occurrence ordering (vs. sorted) is a deliberate choice:
-- | sorted vocabulary order changes if the corpus changes, even when
-- | the new corpus is a superset; first-occurrence order is stable
-- | for prefixes. Either is fine for training; this one round-trips
-- | better in tests.
fromText :: String -> CharTokenizer
fromText text =
  let
    -- Walk left to right, collect unique chars in first-seen order.
    seenStep (Tuple acc seen) c = case Map.lookup c seen of
      Just _  -> Tuple acc seen
      Nothing -> Tuple (Array.snoc acc c) (Map.insert c (Array.length acc) seen)
    Tuple vocab _ = Array.foldl seenStep (Tuple [] Map.empty) (toCharArray text)
    ctoi = Array.foldl
      (\m (Tuple i c) -> Map.insert c i m)
      Map.empty
      (Array.mapWithIndex Tuple vocab)
  in
    CharTokenizer { vocab, ctoi }

-- | Encode a string to a list of token IDs. Characters not in the
-- | tokenizer's vocabulary are dropped silently — caller's choice not
-- | to validate at this layer (the demo always tokenizes from the
-- | same corpus that built the tokenizer, so unknowns shouldn't
-- | happen). If you need strict failure, check `Map.lookup` against
-- | the underlying record yourself.
encode :: CharTokenizer -> String -> Array Int
encode (CharTokenizer t) s = Array.mapMaybe (\c -> Map.lookup c t.ctoi) (toCharArray s)

-- | Decode a list of token IDs back to a string. IDs out of range are
-- | dropped. Round-trips with `encode` for any input that's a subset
-- | of the corpus the tokenizer was built from.
decode :: CharTokenizer -> Array Int -> String
decode (CharTokenizer t) ids =
  fromCharArray (Array.mapMaybe (\i -> Array.index t.vocab i) ids)

-- | Vocabulary size (= number of distinct characters in the corpus).
-- | Equal to the `vocab_size` you'd pass to a transformer's
-- | embedding/lm-head dimensions.
size :: CharTokenizer -> Int
size (CharTokenizer t) = Array.length t.vocab

-- | The vocabulary as an array (index → char). Useful when wiring
-- | sampling code that reports decoded chars one at a time.
chars :: CharTokenizer -> Array Char
chars (CharTokenizer t) = t.vocab
