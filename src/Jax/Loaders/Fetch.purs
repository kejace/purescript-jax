-- | Async fetch helpers with OPFS-backed caching. The JS side returns
-- | a JS Promise that the PS side lifts into `Aff` via `aff-promise`.
-- | This replaces the older callback-style API: `Aff` lets the caller
-- | sequence fetches in flat `do`-notation with normal short-circuiting
-- | error handling, instead of nesting `onOk`/`onErr` continuations.
module Jax.Loaders.Fetch
  ( fetchBytes
  , fetchText
  , fetchTextLines
  -- * Legacy callback API (kept for any caller that hasn't migrated)
  , fetchBytesCb
  , fetchTextCb
  ) where

import Prelude

import Control.Promise (Promise, toAffE)
import Data.Array (filter)
import Data.String (Pattern(..), split)
import Effect (Effect)
import Effect.Aff (Aff)
import Effect.Uncurried (EffectFn1, EffectFn3, runEffectFn1, runEffectFn3)
import Foreign (Foreign)

foreign import fetchBytesAffImpl :: EffectFn1 String (Promise Foreign)
foreign import fetchTextAffImpl :: EffectFn1 String (Promise String)

-- | Fetch a URL and return its body as a `Foreign` (a `Uint8Array`).
-- | OPFS-cached on first hit, served from cache on subsequent requests.
-- | Throws via Aff's error channel on HTTP errors / network failures.
fetchBytes :: String -> Aff Foreign
fetchBytes url = toAffE (runEffectFn1 fetchBytesAffImpl url)

-- | UTF-8 text variant of `fetchBytes`.
fetchText :: String -> Aff String
fetchText url = toAffE (runEffectFn1 fetchTextAffImpl url)

-- | Fetch a UTF-8 text file and split on newlines, dropping empty
-- | lines. Convenient for plain-text corpora (one item per line —
-- | names lists, single-token-per-line vocabularies, simple datasets).
-- | Splits on `\n`; if your file is `\r\n`-terminated, trim outside.
fetchTextLines :: String -> Aff (Array String)
fetchTextLines url = do
  txt <- fetchText url
  pure (filter (_ /= "") (split (Pattern "\n") txt))

-- Legacy callback API ------------------------------------------------

foreign import fetchBytesImpl
  :: EffectFn3 String (Foreign -> Effect Unit) (String -> Effect Unit) Unit

foreign import fetchTextImpl
  :: EffectFn3 String (String -> Effect Unit) (String -> Effect Unit) Unit

fetchBytesCb
  :: String
  -> (Foreign -> Effect Unit)
  -> (String -> Effect Unit)
  -> Effect Unit
fetchBytesCb = runEffectFn3 fetchBytesImpl

fetchTextCb
  :: String
  -> (String -> Effect Unit)
  -> (String -> Effect Unit)
  -> Effect Unit
fetchTextCb = runEffectFn3 fetchTextImpl
