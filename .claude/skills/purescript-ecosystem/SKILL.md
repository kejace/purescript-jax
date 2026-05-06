---
name: purescript-ecosystem
description: PureScript package-ecosystem guide — which library to reach for (HTTP, JSON codecs, web framework, testing, parsing, Node bindings, Web APIs, CLI), with named decision trees and per-package summaries. Use when picking a dependency, evaluating a third-party package, or wiring spago.yaml. Companion to the `purescript` skill (which is for language-level idioms).
---

# PureScript Ecosystem Guide

> AI-to-AI reference for the PureScript package ecosystem. Use this when helping developers write PureScript. For language-level idioms, patterns, and pitfalls, see the companion `/purescript` skill.

## Decision Trees

### "I need to make an HTTP request"
- **Browser**: `affjax-web` (high-level) or `web-fetch` (low-level Fetch API)
- **Node.js**: `affjax-node` or `node-http`
- **Either platform**: `affjax` (core) + platform-specific driver
- **Don't use**: `milkis` (outdated), `fetch` (low-level)

### "I need to parse/produce JSON"
- **Best practice (codec values)**: `codec-argonaut` (Argonaut backend) or `codec-json` (standalone)
- **Record-based auto-derivation**: `yoga-json` (modern) or `simple-json` (older, still works)
- **Low-level Argonaut**: `argonaut-codecs` + `argonaut-core` (manual encoders/decoders)
- **Generic derivation**: `argonaut-generic` (if you must, but prefer codecs)
- **Don't**: Write `EncodeJson`/`DecodeJson` instances directly

### "I need a web framework"
- **Component-based (recommended)**: `halogen` — the standard. Production-ready, excellent tooling.
- **React interop**: `react-basic-hooks` + `react-basic-dom` — when you need React ecosystem.
- **Signals/FRP**: `deku` — experimental, high-performance signal-based UI.
- **Don't use**: `thermite` (dead), `flame` (niche), `concur-react` (experimental)

### "I need an HTTP server"
- **Modern**: `httpurple` — successor to HTTPure, clean routing DSL, middleware support
- **Express.js wrapper**: `express` — if you need Express middleware ecosystem
- **Low-level**: `node-http` — direct Node.js HTTP bindings
- **Don't use**: `httpure` directly (use `httpurple` instead)

### "I need routing"
- **Bidirectional (recommended)**: `routing-duplex` — one definition generates both parser and printer
- **Parser only**: `routing` — applicative URL parser

### "I need to test"
- **Unit/integration**: `spec` — BDD-style, the standard test framework
- **Property-based**: `quickcheck` — Haskell-style property testing
- **Both**: `spec-quickcheck` — run QuickCheck properties inside Spec
- **Test discovery**: `spec-discovery` — auto-discovers test modules

### "I need to work with optional/nullable JS values"
- **JS null/undefined**: `nullable` — `Nullable a` with `toMaybe`/`toNullable`
- **PureScript optionality**: `maybe` — the standard `Maybe` type
- **Don't**: Use raw `Foreign` for simple nullable cases

### "I need database access"
- **PostgreSQL**: `postgresql` — typed PostgreSQL bindings
- **SQLite**: `node-sqlite3` — SQLite3 bindings for Node.js

### "I need to parse text"
- **General parsing**: `parsing` — monadic parser combinators (like Parsec)
- **Simple string parsing**: `string-parsers` — lighter-weight string parsers
- **PureScript source**: `language-cst-parser` — full PureScript CST parser

### "I need optics (lenses/prisms)"
- **Use**: `profunctor-lenses` — profunctor-based optics (Lens', Prism', Traversal', etc.)
- **Don't**: Try to use Haskell's `lens` or `optics` — they don't exist in PureScript

### "I need CLI argument parsing"
- **Declarative**: `optparse` — port of Haskell's optparse-applicative
- **Simple**: `argparse-basic` — lighter alternative by natefaubion

---

## Package Guide by Category

### Core Language

#### prelude
**What:** The PureScript Prelude — basic type classes and functions imported by virtually everything.
**Key exports:** `class Show`, `class Eq`, `class Ord`, `class Semigroup`, `class Monoid`, `class Functor`, `class Apply`, `class Applicative`, `class Bind`, `class Monad`, `map`/`<$>`, `apply`/`<*>`, `bind`/`>>=`, `pure`, `unit`, `void`, `($)`, `(#)`, `(<<<)`, `(>>>)`
**Note:** PureScript's Prelude is explicit — you must import it. `(#)` is reverse application (like Haskell's `&`). `(<<<)` is function composition (like Haskell's `(.)`).
**Ecosystem role:** core | **Status:** active

#### effect
**What:** The `Effect` monad for synchronous side effects.
**Key exports:** `Effect`, `class MonadEffect`, `liftEffect`
**Instead of:** Never use `IO` — PureScript has no `IO` type.
**Note:** `Effect` is PureScript's equivalent of Haskell's `IO` for sync effects only. For async, use `Aff`.
**Ecosystem role:** core | **Status:** active

#### control
**What:** Control flow abstractions — Alternative, MonadPlus, MonadZero, etc.
**Key exports:** `class Alternative`, `class MonadPlus`, `guard`, `when`, `unless`, `ifM`, `class Comonad`, `extend`, `extract`
**Note:** `when`/`unless` live here. Always prefer `when cond action` over `if cond then action else pure unit`.
**Ecosystem role:** core | **Status:** active

#### transformers
**What:** Monad transformers — StateT, ReaderT, WriterT, ExceptT, MaybeT, etc.
**Key exports:** `StateT`, `ReaderT`, `WriterT`, `ExceptT`, `MaybeT`, `runStateT`, `runReaderT`, `runWriterT`, `runExceptT`, `lift`, `class MonadTrans`, `class MonadState`, `class MonadReader`, `class MonadWriter`, `class MonadThrow`, `class MonadError`
**Note:** MTL-style classes are in this package (not a separate `mtl` package like Haskell). `ExceptT` replaces `EitherT`.
**Ecosystem role:** core | **Status:** active

#### aff
**What:** Asynchronous effect monad with cancellation, error handling, and fiber support.
**Key exports:** `Aff`, `class MonadAff`, `liftAff`, `launchAff_`, `launchAff`, `forkAff`, `joinFiber`, `killFiber`, `delay`, `attempt`, `throwError`, `catchError`, `bracket`, `Milliseconds`
**When to use:** Any async operation — HTTP requests, timers, file I/O, database queries.
**Instead of:** Never use callbacks or raw Promise-based APIs when `Aff` is available.
**Note:** `launchAff_` runs an `Aff` from `Effect` (discarding the fiber). `launchAff` returns the fiber. Use `bracket` for resource cleanup. Use `parallel` package for concurrent `Aff` operations.
**Ecosystem role:** core (contrib org) | **Status:** active

#### parallel
**What:** `MonadPar` abstraction for running effects in parallel.
**Key exports:** `class Parallel`, `parallel`, `sequential`, `parTraverse`, `parTraverse_`, `parSequence`
**When to use:** Running multiple `Aff` computations concurrently.
**Note:** `parTraverse` is the parallel equivalent of `traverse`. Works with `Aff` out of the box.
**Ecosystem role:** core | **Status:** active

#### exceptions
**What:** Exception handling for `Effect` and `Aff`.
**Key exports:** `class MonadThrow`, `class MonadError`, `throw`, `catchException`, `try`, `Error`, `error`, `message`
**Note:** JavaScript exceptions are `Error` (from `Effect.Exception`). For typed errors, use `ExceptT` from `transformers`.
**Ecosystem role:** core | **Status:** active

#### refs
**What:** Mutable references in `Effect`.
**Key exports:** `Ref`, `new`, `read`, `write`, `modify`, `modify'`
**Note:** Like Haskell's `IORef`. For `Aff`, use `AVar` from the `avar` package instead.
**Ecosystem role:** core | **Status:** active

#### st
**What:** Safe local mutation via the ST monad.
**Key exports:** `ST`, `STRef`, `run`, `new`, `read`, `write`, `modify`
**When to use:** Performance-critical code that needs local mutation without escaping.
**Note:** Like Haskell's `ST`. The region parameter ensures refs don't escape.
**Ecosystem role:** core | **Status:** active

#### tailrec
**What:** Stack-safe monadic recursion.
**Key exports:** `tailRec`, `tailRecM`, `Step(..)`, `Done`, `Loop`
**When to use:** When you need guaranteed stack safety in recursive monadic code.
**Note:** Return `Loop` to continue, `Done` to finish. The compiler also does TCO for self-recursive functions, but `tailRecM` works for monadic recursion.
**Ecosystem role:** core | **Status:** active

#### lazy
**What:** Lazy evaluation support.
**Key exports:** `Lazy`, `defer`, `force`
**Note:** PureScript is strict by default. Use `Lazy` explicitly when you need laziness.
**Ecosystem role:** core | **Status:** active

#### debug
**What:** Debug tracing (console.log for debugging).
**Key exports:** `spy`, `trace`, `traceM`, `debugger`
**When to use:** Debugging. Remove before committing.
**Instead of:** Don't use `Console.log` for debugging — `spy` and `trace` work in any context.
**Note:** By garyb. `spy` logs and returns the value. `trace` takes a message and a thunk. `debugger` inserts a JS `debugger` statement.
**Ecosystem role:** community | **Status:** active

#### console
**What:** Console output functions.
**Key exports:** `log`, `warn`, `error`, `info`, `time`, `timeEnd`
**When to use:** Actual logging output (not debugging — use `debug` for that).
**Ecosystem role:** core | **Status:** active

#### functions
**What:** Utility functions.
**Key exports:** `const`, `flip`, `apply`, `on`, `applyN`
**Ecosystem role:** core | **Status:** active

#### partial
**What:** Utilities for partial functions.
**Key exports:** `Partial`, `crashWith`, `unsafePartial`, `unsafeCrashWith`
**Note:** `Partial` is a compiler-solved constraint. Use `unsafePartial` to discharge it. Avoid where possible.
**Ecosystem role:** core | **Status:** active

#### unsafe-coerce
**What:** Unsafe type coercion (`unsafeCoerce :: forall a b. a -> b`).
**Key exports:** `unsafeCoerce`
**When to use:** FFI interop or performance-critical code where you can guarantee type safety externally. Very widely depended upon (162 reverse deps) because many FFI bindings use it internally.
**Instead of:** Don't use for application logic — use `Newtype` unwrap/wrap or proper type conversions.
**Ecosystem role:** core | **Status:** active

#### safe-coerce
**What:** Type-safe coercion between representationally equal types.
**Key exports:** `coerce`, `class Coercible`
**When to use:** Converting between newtypes and their underlying types without `unsafeCoerce`. Compiler-checked.
**Note:** Like Haskell's `Coercible`. Prefer this over `unsafeCoerce` when the types are representationally equal.
**Ecosystem role:** core | **Status:** active

#### type-equality
**What:** Type equality evidence.
**Key exports:** `class TypeEquals`, `proof`, `from`, `to`
**When to use:** Type-level programming where you need to witness that two types are the same.
**Ecosystem role:** core | **Status:** active

#### contravariant
**What:** Contravariant functors.
**Key exports:** `class Contravariant`, `cmap`, `(>$<)`, `Predicate`, `Comparison`, `Equivalence`, `Op`
**When to use:** Types that consume values rather than produce them (predicates, comparisons, serializers).
**Ecosystem role:** core | **Status:** active

#### exists
**What:** Existential types.
**Key exports:** `Exists`, `mkExists`, `runExists`
**When to use:** Hiding a type parameter — e.g., heterogeneous collections.
**Ecosystem role:** core | **Status:** active

#### random
**What:** Random number generation.
**Key exports:** `random`, `randomInt`, `randomRange`, `randomBool`
**Note:** Uses `Effect` — wraps `Math.random()`. For reproducible randomness, use `lcg` or `gen` + `quickcheck`.
**Ecosystem role:** core | **Status:** active

### Data Types & Structures

#### maybe
**What:** The `Maybe` type for optional values.
**Key exports:** `Maybe(..)`, `Just`, `Nothing`, `maybe`, `fromMaybe`, `fromMaybe'`, `isJust`, `isNothing`, `optional`
**Ecosystem role:** core | **Status:** active

#### either
**What:** The `Either` type for computations that can fail.
**Key exports:** `Either(..)`, `Left`, `Right`, `either`, `fromLeft`, `fromRight`, `note`, `hush`
**Note:** `note` converts `Maybe` to `Either` (adds an error). `hush` converts `Either` to `Maybe` (discards the error).
**Ecosystem role:** core | **Status:** active

#### tuples
**What:** Tuple types.
**Key exports:** `Tuple(..)`, `fst`, `snd`, `curry`, `uncurry`, `swap`
**Note:** PureScript has no built-in tuple syntax. Use `Tuple a b` or `/\` operator. For larger tuples, use records instead.
**Ecosystem role:** core | **Status:** active

#### arrays
**What:** Immutable arrays (JavaScript arrays under the hood).
**Key exports:** `Array`, `cons`, `snoc`, `head`, `tail`, `last`, `init`, `index`/`(!!)`, `filter`, `sort`, `sortBy`, `nub`, `nubEq`, `zip`, `unzip`, `concat`, `concatMap`, `mapWithIndex`, `foldl`, `foldr`, `length`, `null`, `elem`, `find`, `findIndex`
**Note:** Arrays are the primary sequential collection. Use `lists` only when you specifically need cons-based linked lists. PureScript arrays are JavaScript arrays — O(1) index, O(n) cons.
**Ecosystem role:** core | **Status:** active

#### lists
**What:** Strict and lazy linked lists.
**Key exports:** `List(..)`, `Cons`, `Nil`, `NonEmptyList`
**When to use:** When you need efficient cons/uncons or pattern matching on head/tail. Rare in practice — prefer `Array` for most use cases.
**Ecosystem role:** core | **Status:** active

#### ordered-collections
**What:** Map and Set implemented as balanced trees.
**Key exports:** `Map`, `Set`, `Map.empty`, `Map.insert`, `Map.lookup`, `Map.delete`, `Map.fromFoldable`, `Map.toUnfoldable`, `Set.empty`, `Set.insert`, `Set.member`
**Note:** Keys must have `Ord` instance. These are persistent/immutable. For mutable or hash-based collections, see `unordered-collections`.
**Ecosystem role:** core | **Status:** active

#### strings
**What:** String manipulation functions.
**Key exports:** `length`, `charAt`, `toCharArray`, `fromCharArray`, `indexOf`, `split`, `joinWith`, `replace`, `replaceAll`, `trim`, `toLower`, `toUpper`, `take`, `drop`, `Pattern(..)`
**Note:** PureScript strings are JavaScript strings (UTF-16). `Pattern` newtype for search patterns. `CodePoint` and `CodeUnit` modules for Unicode-correct operations.
**Ecosystem role:** core | **Status:** active

#### integers
**What:** Integer type and operations.
**Key exports:** `Int`, `fromNumber`, `toNumber`, `fromString`, `floor`, `ceil`, `round`, `even`, `odd`
**Note:** PureScript `Int` is a 32-bit signed integer (JavaScript number restricted to integer range).
**Ecosystem role:** core | **Status:** active

#### numbers
**What:** Number type utilities.
**Key exports:** `Number`, `fromString`, `nan`, `isNaN`, `infinity`, `isFinite`
**Note:** PureScript `Number` is a JavaScript 64-bit float.
**Ecosystem role:** core | **Status:** active

#### newtype
**What:** Type class and utilities for newtypes.
**Key exports:** `class Newtype`, `unwrap`, `wrap`, `over`, `under`, `ala`, `alaF`
**Note:** Derive with `derive instance Newtype MyType _`. Use `unwrap`/`wrap` instead of manual pattern matching. `ala` and `alaF` for newtype-directed folding (e.g., `ala Additive foldMap`).
**Ecosystem role:** core | **Status:** active

#### foreign
**What:** Working with untyped JavaScript values.
**Key exports:** `Foreign`, `unsafeToForeign`, `unsafeFromForeign`, `readString`, `readNumber`, `readBoolean`, `readArray`, `readNull`, `readUndefined`, `typeOf`, `tagOf`
**When to use:** Low-level FFI interop. Prefer typed approaches (`codec-argonaut`, `yoga-json`) when possible.
**Ecosystem role:** core | **Status:** active

#### foreign-object
**What:** Untyped JavaScript objects as string-keyed maps.
**Key exports:** `Object`, `empty`, `insert`, `lookup`, `delete`, `keys`, `values`, `fromFoldable`, `toUnfoldable`, `singleton`
**When to use:** Interfacing with JS code that passes plain objects, or when you need string-keyed maps backed by JS objects rather than balanced trees.
**Ecosystem role:** core | **Status:** active

#### nullable
**What:** Working with JavaScript null values.
**Key exports:** `Nullable`, `toMaybe`, `toNullable`, `null`, `notNull`
**When to use:** FFI boundaries where JS functions return or accept null.
**Instead of:** Don't use `Foreign` directly for simple nullable values.
**Ecosystem role:** contrib | **Status:** active

#### nonempty
**What:** Non-empty collections.
**Key exports:** `NonEmpty(..)`, `NonEmptyArray`, `NonEmptyList`, `fromArray`, `fromFoldable`, `head`, `tail`, `singleton`
**When to use:** When you can guarantee a collection has at least one element — encodes this in the type.
**Ecosystem role:** core | **Status:** active

#### variant
**What:** Polymorphic variants — extensible sum types using row types.
**Key exports:** `Variant`, `inj`, `on`, `onMatch`, `case_`, `match`, `default`, `expand`, `contract`
**When to use:** When you need open/extensible sum types, or to combine effect rows. Core building block of `Run`.
**Instead of:** Don't define many small ADTs when you need extensibility — use `Variant`.
**Note:** By natefaubion. Uses `Symbol` and row types to create type-safe extensible variants. `match` takes a record of handlers.
**Ecosystem role:** community | **Status:** active

#### validation
**What:** Applicative validation that accumulates errors.
**Key exports:** `V`, `invalid`, `validation`, `toEither`, `fromEither`
**When to use:** Form validation or any scenario where you want ALL errors, not just the first.
**Instead of:** Don't use `Either` when you need to accumulate multiple errors — `Either` short-circuits.
**Note:** `V err a` has `Applicative` that accumulates `err` (requires `Semigroup err`). No `Monad` instance by design.
**Ecosystem role:** core | **Status:** active

#### datetime
**What:** Date and time types and operations.
**Key exports:** `DateTime`, `Date`, `Time`, `Year`, `Month`, `Day`, `Hour`, `Minute`, `Second`, `Millisecond`, `Weekday`, `diff`, `adjust`, `date`, `time`
**Note:** Pure date/time types. For JS Date interop, use `js-date`. For precise timestamps, see `precise-datetime`.
**Ecosystem role:** core | **Status:** active

#### enums
**What:** Operations on enums (types with `Bounded` + `Enum`).
**Key exports:** `class Enum`, `class BoundedEnum`, `succ`, `pred`, `toEnum`, `fromEnum`, `enumFromTo`, `upFromIncluding`, `downFromIncluding`
**Ecosystem role:** core | **Status:** active

#### record
**What:** Record manipulation utilities.
**Key exports:** `merge`, `union`, `disjointUnion`, `nub`, `delete`, `rename`, `get`, `set`, `modify`, `insert`, `class EqualFields`
**Note:** Works with PureScript's structural record types and row polymorphism. Type-level label manipulation.
**Ecosystem role:** core | **Status:** active

#### these
**What:** The `These` type — inclusive disjunction (`This a`, `That b`, `Both a b`).
**Key exports:** `These(..)`, `these`, `thisOrBoth`, `thatOrBoth`, `fromThese`
**When to use:** When you need to represent "one, the other, or both" — e.g., merging two data sources.
**Ecosystem role:** contrib | **Status:** active

#### tree-rose
**What:** Rose trees (multi-way trees).
**Key exports:** `Tree(..)`, `Forest`, `mkTree`, `drawTree`, `scanTree`
**When to use:** Hierarchical/tree-structured data. DOM trees, ASTs, file trees.
**Ecosystem role:** contrib | **Status:** active

#### catenable-lists
**What:** Catenable lists — O(1) append.
**Key exports:** `CatList`, `CatQueue`, `empty`, `cons`, `snoc`, `append`, `uncons`
**When to use:** When you need efficient repeated appending (e.g., building output).
**Ecosystem role:** core | **Status:** active

#### untagged-union
**What:** Untagged unions for JS interop — represent `string | number | null` etc.
**Key exports:** `OneOf`, `asOneOf`, `fromOneOf`, `type (|+|)`
**When to use:** FFI where JS functions accept or return untagged union types.
**Note:** By rowtype-yoga. Encodes JS union types at the PureScript type level.
**Ecosystem role:** community | **Status:** active

#### literals
**What:** Type-level literal values (strings, ints, booleans) reflected to term level.
**Key exports:** `StringLit`, `IntLit`, `BooleanLit`, `literal`
**When to use:** Type-safe config, record keys as type-level strings.
**Note:** By rowtype-yoga. Works with `Symbol` proxy patterns.
**Ecosystem role:** community | **Status:** active

#### record-studio
**What:** Advanced record manipulation utilities.
**Key exports:** Record merging, key mapping, value mapping, optional fields.
**When to use:** Complex record transformations beyond what `record` provides.
**Note:** By rowtype-yoga.
**Ecosystem role:** community | **Status:** active

### Type Classes & Abstractions

#### foldable-traversable
**What:** `Foldable` and `Traversable` type classes and utilities.
**Key exports:** `class Foldable`, `class Traversable`, `foldMap`, `fold`, `foldl`, `foldr`, `traverse`, `sequence`, `for`, `for_`, `traverse_`, `any`, `all`, `find`, `elem`, `sum`, `product`, `maximum`, `minimum`, `length`, `null`, `intercalate`
**Note:** `for` is `traverse` with arguments flipped. `for_`/`traverse_` discard results. This is extremely heavily used.
**Ecosystem role:** core | **Status:** active

#### bifunctors
**What:** Bifunctor, Biapplicative, and related classes.
**Key exports:** `class Bifunctor`, `bimap`, `lmap`, `rmap`
**When to use:** Working with two-parameter types like `Either` or `Tuple`. `lmap` maps over the left, `rmap` over the right.
**Ecosystem role:** core | **Status:** active

#### profunctor
**What:** Profunctor type class and instances.
**Key exports:** `class Profunctor`, `dimap`, `lcmap`, `rmap`, `class Strong`, `class Choice`, `class Closed`
**When to use:** Advanced abstractions. Foundation for profunctor-lenses.
**Ecosystem role:** core | **Status:** active

#### profunctor-lenses
**What:** Profunctor-based optics library.
**Key exports:** `Lens`, `Lens'`, `Prism`, `Prism'`, `Traversal`, `Traversal'`, `Iso`, `Iso'`, `lens`, `prism`, `prism'`, `_Just`, `_Nothing`, `_Left`, `_Right`, `view`, `preview`, `set`, `over`, `(^.)`, `(.~)`, `(%~)`, `(^?)`, `_1`, `_2`
**When to use:** Composable accessors/modifiers for nested data structures.
**Note:** Operator style: `record ^. _field`, `record # _field .~ newVal`. Composes with `(<<<)`.
**Ecosystem role:** contrib | **Status:** active

#### free
**What:** Free monads and free applicatives.
**Key exports:** `Free`, `liftF`, `foldFree`, `runFree`, `runFreeM`, `resume`, `FreeAp`, `liftFreeAp`, `foldFreeAp`
**When to use:** Building DSLs as data. Natural transformation (`foldFree`) to interpret. Consider `Run` as a modern alternative.
**Ecosystem role:** core | **Status:** active

#### run
**What:** Extensible effects using free monads + variants.
**Key exports:** `Run`, `lift`, `extract`, `interpret`, `run`, `runBaseAff`, `runBaseEffect`, `EFFECT`, `AFF`, `EXCEPT`, `STATE`, `READER`
**When to use:** When you need composable, extensible effect systems. Each effect is a functor row, interpreted independently.
**Instead of:** Deeply nested monad transformer stacks.
**Note:** By natefaubion. More ergonomic than raw `Free` for multi-effect scenarios. Effects compose via row types.
**Ecosystem role:** community | **Status:** active

#### typelevel-prelude
**What:** Type-level programming utilities.
**Key exports:** `class IsSymbol`, `class TypeEquals`, `class RowToList`, `RLProxy`, `Proxy`, `SProxy`
**Note:** Foundation for type-level record programming. `RowToList` converts row types to type-level lists for iteration.
**Ecosystem role:** core | **Status:** active

#### heterogeneous
**What:** Heterogeneous maps and folds over records.
**Key exports:** `class HMap`, `class HFoldl`, `class HFoldr`, `class Mapping`, `hmap`, `hfoldl`
**When to use:** When you need to map or fold a function over all fields of a record, regardless of their types.
**Note:** By natefaubion. Powerful but advanced — type-level programming required.
**Ecosystem role:** community | **Status:** active

#### filterable
**What:** `Filterable` and `Witherable` type classes.
**Key exports:** `class Filterable`, `class Witherable`, `filter`, `filterMap`, `partition`, `partitionMap`, `wither`, `wilt`
**When to use:** `filterMap` is like `mapMaybe` from Haskell — maps and filters in one pass.
**Ecosystem role:** core | **Status:** active

#### distributive
**What:** `Distributive` type class (dual of `Traversable`).
**Key exports:** `class Distributive`, `distribute`, `cotraverse`, `collect`
**Ecosystem role:** core | **Status:** active

#### invariant
**What:** `Invariant` functor — both covariant and contravariant.
**Key exports:** `class Invariant`, `imap`
**When to use:** Types that both consume and produce values (codecs, lenses).
**Ecosystem role:** core | **Status:** active

#### functors
**What:** Functor products, coproducts, and composition.
**Key exports:** `Product`, `Coproduct`, `Compose`, `FunctorWithIndex`, `FoldableWithIndex`, `TraversableWithIndex`, `mapWithIndex`, `foldMapWithIndex`, `traverseWithIndex`
**When to use:** The `*WithIndex` classes are the practical draw — indexed traversals over arrays, maps, etc.
**Ecosystem role:** core | **Status:** active

#### gen
**What:** QuickCheck generator combinators.
**Key exports:** `Gen`, `chooseInt`, `chooseFloat`, `oneOf`, `frequency`, `elements`, `unfoldable`, `suchThat`, `resize`, `sized`
**When to use:** Building custom `Arbitrary` generators for property testing.
**Note:** Separated from `quickcheck` so generators can be used independently.
**Ecosystem role:** core | **Status:** active

#### freet
**What:** Free monad transformer.
**Key exports:** `FreeT`, `liftFreeT`, `foldFreeT`, `interpret`, `resume`
**When to use:** Free monad DSLs that need to be layered on another monad (e.g., `Aff`).
**Ecosystem role:** contrib | **Status:** active

#### avar
**What:** Asynchronous mutable variables (like Haskell's MVar).
**Key exports:** `AVar`, `new`, `empty`, `take`, `put`, `read`, `tryTake`, `tryPut`, `tryRead`, `kill`, `status`
**When to use:** Coordination between concurrent `Aff` fibers. Also used internally by `Aff`.
**Note:** `empty` creates a blocking var. `new` creates a filled var. `take` blocks until a value is available.
**Ecosystem role:** contrib | **Status:** active

#### fork
**What:** Utilities for forking computations.
**Key exports:** `bracket`, `finally`, `fork`, `suspend`
**When to use:** Resource management — `bracket` ensures cleanup even on exceptions.
**Ecosystem role:** contrib | **Status:** active

#### qualified-do
**What:** Use different `bind`/`apply`/`discard`/`pure` in `do` and `ado` blocks via qualified syntax.
**Key exports:** Documentation/pattern only — enables `MyModule.do` and `MyModule.ado` syntax.
**When to use:** Using applicative `ado` with custom applicatives, or indexed monads with `do`.
**Note:** By artemis-prime. PureScript supports qualified `do`/`ado` natively; this package provides the convention.
**Ecosystem role:** community | **Status:** active

### JSON & Serialization

#### argonaut-core
**What:** Core JSON types for the Argonaut ecosystem.
**Key exports:** `Json`, `jsonNull`, `jsonTrue`, `jsonFalse`, `fromString`, `fromNumber`, `fromObject`, `fromArray`, `caseJson`, `caseJsonNull`, `caseJsonBoolean`, `caseJsonNumber`, `caseJsonString`, `caseJsonArray`, `caseJsonObject`, `stringify`, `stringifyWithIndent`
**Note:** Low-level JSON AST. Most code should use `codec-argonaut` on top of this.
**Ecosystem role:** contrib | **Status:** active

#### argonaut-codecs
**What:** Encode/decode type classes for Argonaut.
**Key exports:** `class EncodeJson`, `class DecodeJson`, `encodeJson`, `decodeJson`, `JsonDecodeError(..)`, `printJsonDecodeError`
**When to use:** If the project convention is type-class-based JSON. But prefer codec values.
**Note:** These are the instances that `codec-argonaut` replaces with explicit codec values.
**Ecosystem role:** contrib | **Status:** active

#### argonaut
**What:** Umbrella package re-exporting argonaut-core, argonaut-codecs, argonaut-traversals.
**Ecosystem role:** contrib | **Status:** active

#### argonaut-generic
**What:** Generic JSON encoding/decoding for Argonaut.
**Key exports:** `genericEncodeJson`, `genericDecodeJson`
**When to use:** Quick-and-dirty JSON for ADTs. The encoding format may not match your API.
**Instead of:** For API boundaries, prefer explicit codecs for control over the format.
**Ecosystem role:** contrib | **Status:** active

#### codec
**What:** Core codec abstraction — bidirectional, composable encoders/decoders.
**Key exports:** `Codec`, `Codec'`, `BasicCodec`, `decode`, `encode`, `mapCodec`, `composeCodec`, `(~)`, `(<~<)`, `(>~>)`
**When to use:** Foundation for `codec-argonaut` and `codec-json`. You typically import the backend package directly.
**Note:** By garyb. A `Codec` carries both encoding and decoding in one value. Profunctor-based.
**Ecosystem role:** community | **Status:** active

#### codec-argonaut
**What:** JSON codecs using Argonaut as the backend.
**Key exports:** `JsonCodec`, `object`, `record`, `recordProp`, `string`, `number`, `int`, `boolean`, `array`, `null`, `json`, `named`, `prismaticCodec`, `coercible`, `sum`, `taggedSum`, `fix`, `decode`, `encode`, `printJsonDecodeError`
**When to use:** This is the recommended approach for JSON in PureScript. Define codec values, use them explicitly.
**Example pattern:**
```purescript
import Data.Codec.Argonaut as CA
import Data.Codec.Argonaut.Record as CAR

userCodec :: JsonCodec User
userCodec = CA.object "User" $ CAR.record
  { name: CA.string
  , age: CA.int
  , email: CA.string
  }
```
**Instead of:** Type class instances (`EncodeJson`/`DecodeJson`), which create orphan instance problems and invisible behavior.
**Note:** By garyb. `CAR.record` is the most common entry point — it derives a codec from a record of field codecs.
**Ecosystem role:** community | **Status:** active

#### codec-json
**What:** JSON codecs using a standalone JSON backend (no Argonaut dependency).
**Key exports:** Similar API to `codec-argonaut` but against a different JSON type.
**When to use:** If you want codecs without the Argonaut dependency.
**Note:** By garyb. Newer alternative to `codec-argonaut`.
**Ecosystem role:** community | **Status:** active

#### simple-json
**What:** Auto-derived JSON encoding/decoding via row types.
**Key exports:** `readJSON`, `writeJSON`, `readJSON'`, `class ReadForeign`, `class WriteForeign`
**When to use:** When you want zero-boilerplate JSON for records that match JS object shapes exactly.
**Instead of:** Writing manual codecs when the shape matches 1:1.
**Note:** By justinwoo. Uses `Foreign` + row types. Great for rapid prototyping. Less control than codecs.
**Ecosystem role:** community | **Status:** active

#### yoga-json
**What:** Modern JSON encoding/decoding with better error messages than simple-json.
**Key exports:** `readJSON`, `writeJSON`, `class ReadForeign`, `class WriteForeign`, `(:::)`, `writeJSONImpl`, `readJSONImpl`
**When to use:** Modern alternative to `simple-json` with better error reporting and more features.
**Note:** By rowtype-yoga. Spiritual successor to simple-json.
**Ecosystem role:** community | **Status:** active

### Web Frameworks

#### halogen
**What:** Type-safe, component-based UI framework. The most popular PureScript web framework.
**Key exports:** `Component`, `ComponentHTML`, `ComponentSpec`, `mkComponent`, `HalogenM`, `raise`, `query`, `tell`, `Slot`, `slot`, `slot_`
**When to use:** Building web applications. The default choice for PureScript web UIs.
**Note:** Component model with `State`, `Action`, `Input`, `Output`, `Query`, `Slots`. Uses a virtual DOM (`halogen-vdom`). Components communicate via output messages (up) and queries (down). `HalogenM` monad for component logic.
**Key patterns:**
- `handleAction :: Action -> HalogenM ...` — handles events
- `render :: State -> ComponentHTML Action Slots Output m` — renders HTML
- `initialState :: Input -> State`
- `H.liftAff` / `H.liftEffect` for effects inside `HalogenM`
**Ecosystem role:** community (purescript-halogen org) | **Status:** active

#### halogen-hooks
**What:** React-style hooks for Halogen components.
**Key exports:** `useState`, `useQuery`, `useLifecycleEffect`, `useTickEffect`, `Hook`, `component`, `bind`, `discard`
**When to use:** When you prefer a hooks-based API over Halogen's traditional component model.
**Note:** By thomashoneyman.
**Ecosystem role:** community | **Status:** active

#### halogen-store
**What:** Global state management for Halogen.
**Key exports:** `Store`, `mkStore`, `connect`, `selectAll`, `selectEq`
**When to use:** Shared state across multiple Halogen components (like Redux for Halogen).
**Note:** By thomashoneyman.
**Ecosystem role:** community | **Status:** active

#### halogen-formless
**What:** Type-safe form handling for Halogen.
**Key exports:** `FormlessAction`, `FormField`, `mkFormSpec`, validation combinators
**When to use:** Complex forms with validation in Halogen apps.
**Note:** By thomashoneyman.
**Ecosystem role:** community | **Status:** active

#### halogen-subscriptions
**What:** Event source subscriptions for Halogen.
**Key exports:** `Emitter`, `Listener`, `subscribe`, `create`, `notify`, `makeEmitter`
**When to use:** Hooking external event sources (WebSocket, timers, DOM events) into Halogen components.
**Ecosystem role:** community | **Status:** active

#### react-basic
**What:** Core React bindings — base types shared by all react-basic-* packages.
**Key exports:** `JSX`, `ReactComponent`, `element`, `fragment`, `empty`, `keyed`, `provider`, `createContext`
**When to use:** Always needed when using react-basic-hooks or react-basic-dom. Provides the core JSX type.
**Ecosystem role:** community | **Status:** active

#### react-basic-hooks
**What:** React Hooks bindings for PureScript.
**Key exports:** `component`, `useState`, `useEffect`, `useRef`, `useMemo`, `useCallback`, `useContext`, `useReducer`
**When to use:** When you need React (e.g., for React ecosystem interop, existing React codebase).
**Note:** By purescript-react org. Requires `react-basic-dom` for DOM elements.
**Ecosystem role:** community | **Status:** active

#### react-basic-dom
**What:** DOM element bindings for react-basic.
**Key exports:** `R.div`, `R.span`, `R.button`, `R.input`, `R.text`, event handlers, CSS properties
**When to use:** Always paired with `react-basic-hooks`.
**Ecosystem role:** community | **Status:** active

#### deku
**What:** Signal-based UI framework.
**When to use:** Experimental/niche. When you want FRP-style signals instead of component state.
**Note:** By mikesol. Not as widely adopted as Halogen or react-basic.
**Ecosystem role:** community | **Status:** active

### HTTP Client

#### affjax
**What:** HTTP request library returning `Aff`.
**Key exports:** `Request`, `Response`, `defaultRequest`, `request`, `get`, `post`, `put`, `delete`, `printError`
**When to use:** HTTP requests in `Aff`. Needs a platform driver.
**Note:** The core package. Use with `affjax-web` (browser XHR) or `affjax-node` (Node.js HTTP).
**Ecosystem role:** contrib | **Status:** active

#### affjax-web
**What:** Browser (XHR) driver for affjax.
**Key exports:** `driver`
**When to use:** HTTP requests in browser code.
**Example:** `Affjax.request driver { ...request }`
**Ecosystem role:** contrib | **Status:** active

#### affjax-node
**What:** Node.js driver for affjax.
**Key exports:** `driver`
**When to use:** HTTP requests in Node.js code.
**Ecosystem role:** contrib | **Status:** active

#### web-fetch
**What:** Bindings to the Fetch API.
**Key exports:** `fetch`, `Request`, `Response`, `Headers`
**When to use:** Low-level Fetch API access. Prefer `affjax-web` for higher-level usage.
**Ecosystem role:** web | **Status:** active

### HTTP Server

#### httpurple
**What:** Modern HTTP server framework (successor to HTTPure).
**Key exports:** `serve`, `ok`, `badRequest`, `notFound`, `header`, `Route`, `mkRoute`
**When to use:** Building HTTP servers in PureScript/Node.js.
**Instead of:** Old `httpure` package. Use `httpurple` (the renamed/improved fork).
**Note:** Clean routing DSL, middleware support, request/response helpers.
**Ecosystem role:** community | **Status:** active

#### node-http
**What:** Low-level Node.js HTTP bindings.
**Key exports:** `createServer`, `listen`, `Request`, `Response`, `requestMethod`, `requestURL`, `setHeader`, `setStatusCode`, `end`
**When to use:** When you need direct Node.js HTTP access. Prefer `httpurple` for application servers.
**Ecosystem role:** node | **Status:** active

### Routing

#### routing-duplex
**What:** Bidirectional routing — one definition generates both parser and printer.
**Key exports:** `RouteDuplex`, `RouteDuplex'`, `root`, `path`, `segment`, `param`, `optional`, `flag`, `int`, `string`, `end`, `parse`, `print`, `prefix`, `suffix`, `prop`, `record`, `sum`, `generic`
**When to use:** Always prefer this over `routing` — you get URL generation for free.
**Example pattern:**
```purescript
route :: RouteDuplex' MyRoute
route = root $ sum
  { "Home": end
  , "Profile": path "profile" / segment
  , "Search": path "search" $ params { query: string, page: optional <<< int }
  }
```
**Note:** By natefaubion. The `print`/`parse` pair is guaranteed consistent. Works with Halogen routing.
**Ecosystem role:** community | **Status:** active

#### routing
**What:** Applicative URL parser.
**Key exports:** `Match`, `lit`, `str`, `int`, `end`, `match`
**When to use:** If you only need parsing (not printing). Otherwise prefer `routing-duplex`.
**Ecosystem role:** contrib | **Status:** active

### Testing

#### spec
**What:** BDD-style test framework (like Haskell's hspec).
**Key exports:** `Spec`, `describe`, `it`, `pending`, `pending'`, `beforeAll`, `afterAll`, `beforeEach`, `afterEach`, `parallel`
**When to use:** The standard PureScript test framework.
**Note:** Run with `spago test`. Supports `Aff`-based tests by default. Use `spec-quickcheck` to integrate property tests.
**Ecosystem role:** community | **Status:** active

#### quickcheck
**What:** Property-based testing.
**Key exports:** `quickCheck`, `quickCheck'`, `class Arbitrary`, `class Coarbitrary`, `Gen`, `arbitrary`, `elements`, `oneOf`, `frequency`, `arrayOf`, `Result`, `(===)`, `(/==)`, `(<?>)`
**When to use:** Testing properties that should hold for many inputs.
**Note:** Similar to Haskell's QuickCheck. `(===)` for equality assertions with counterexample reporting.
**Ecosystem role:** core | **Status:** active

#### spec-quickcheck
**What:** Run QuickCheck properties inside Spec tests.
**Ecosystem role:** community | **Status:** active

#### spec-discovery
**What:** Auto-discover test modules for spec.
**Key exports:** `discover`
**When to use:** In `Test.Main` to automatically find all `*Spec` modules.
**Ecosystem role:** community | **Status:** active

### Parsing

#### parsing
**What:** Monadic parser combinators (like Parsec/Megaparsec).
**Key exports:** `Parser`, `runParser`, `fail`, `try`, `lookAhead`, `choice`, `option`, `optional`, `many`, `many1`, `sepBy`, `sepBy1`, `between`, `char`, `string`, `satisfy`, `digit`, `letter`, `eof`, `position`
**When to use:** Complex parsing tasks — config files, DSLs, data formats.
**Note:** `try` for backtracking (like Parsec's `try`). Works with both `String` and `StringWithPos`.
**Ecosystem role:** contrib | **Status:** active

#### string-parsers
**What:** Simpler string parser combinators.
**Key exports:** `Parser`, `runParser`, `string`, `char`, `anyChar`, `satisfy`, `many`, `many1`, `regex`
**When to use:** Simpler parsing tasks. Less overhead than `parsing`.
**Ecosystem role:** contrib | **Status:** active

#### language-cst-parser
**What:** Full PureScript CST (Concrete Syntax Tree) parser.
**Key exports:** `parseModule`, `parseExpr`, `parseType`, `parseBinder`
**When to use:** Tooling that needs to parse PureScript source code.
**Note:** By natefaubion. Powers `tidy` (the PureScript formatter) and IDE tooling.
**Ecosystem role:** community | **Status:** active

### Node.js

#### node-fs
**What:** Node.js file system bindings.
**Key exports:** `readTextFile`, `writeTextFile`, `readFile`, `writeFile`, `stat`, `mkdir`, `readdir`, `unlink`, `rename`, `exists`, `chmod`
**Note:** Most functions are in `Node.FS.Aff` (async) and `Node.FS.Sync` (synchronous). Prefer async.
**Ecosystem role:** node | **Status:** active

#### node-buffer
**What:** Node.js Buffer bindings.
**Key exports:** `Buffer`, `fromString`, `toString`, `fromArray`, `toArray`, `size`, `concat`, `slice`, `copy`
**Ecosystem role:** node | **Status:** active

#### node-streams
**What:** Node.js Stream bindings.
**Key exports:** `Readable`, `Writable`, `Duplex`, `Transform`, `pipe`, `read`, `write`, `end`, `onData`, `onEnd`, `onError`
**Ecosystem role:** node | **Status:** active

#### node-path
**What:** Node.js path module bindings.
**Key exports:** `resolve`, `join`, `basename`, `dirname`, `extname`, `relative`, `normalize`, `sep`
**Ecosystem role:** node | **Status:** active

#### node-process
**What:** Node.js process bindings.
**Key exports:** `argv`, `exit`, `cwd`, `getEnv`, `lookupEnv`, `stdin`, `stdout`, `stderr`
**Ecosystem role:** node | **Status:** active

#### node-child-process
**What:** Node.js child_process bindings.
**Key exports:** `exec`, `execSync`, `spawn`, `fork`
**Ecosystem role:** node | **Status:** active

#### node-event-emitter
**What:** Node.js EventEmitter bindings.
**Key exports:** `EventEmitter`, `on`, `once`, `emit`, `removeListener`
**Ecosystem role:** node | **Status:** active

#### node-execa
**What:** Execute shell commands (like execa for Node.js).
**Key exports:** `execa`, `execaSync`, `ExecaResult`, `ExecaError`
**When to use:** Running shell commands with better ergonomics than raw `child-process`.
**Note:** By JordanMartinez.
**Ecosystem role:** community | **Status:** active

#### node-glob-basic
**What:** File glob pattern matching.
**Key exports:** `glob`, `GlobOptions`
**When to use:** Finding files by pattern (e.g., `"src/**/*.purs"`).
**Note:** By natefaubion.
**Ecosystem role:** community | **Status:** active

### Web APIs

#### web-dom
**What:** DOM API bindings.
**Key exports:** `Document`, `Element`, `Node`, `ParentNode`, `querySelector`, `querySelectorAll`, `createElement`, `appendChild`, `removeChild`, `textContent`, `getAttribute`, `setAttribute`
**Ecosystem role:** web | **Status:** active

#### web-html
**What:** HTML element bindings.
**Key exports:** `HTMLElement`, `HTMLInputElement`, `HTMLDocument`, `window`, `document`, `location`, `history`, `fromElement`, `toElement`, `value`, `setValue`
**Ecosystem role:** web | **Status:** active

#### web-events
**What:** DOM Event bindings.
**Key exports:** `Event`, `EventTarget`, `EventType`, `addEventListener`, `removeEventListener`, `preventDefault`, `stopPropagation`, `target`, `currentTarget`
**Ecosystem role:** web | **Status:** active

#### web-storage
**What:** Web Storage API (localStorage/sessionStorage).
**Key exports:** `Storage`, `localStorage`, `sessionStorage`, `getItem`, `setItem`, `removeItem`, `clear`, `length`
**Ecosystem role:** web | **Status:** active

#### web-uievents
**What:** UI event bindings — mouse, keyboard, focus, input events.
**Key exports:** `MouseEvent`, `KeyboardEvent`, `FocusEvent`, `InputEvent`, `WheelEvent`, `TouchEvent`, conversions from base `Event`
**When to use:** Handling specific UI events in DOM code. Halogen wraps these but you need them for raw DOM interop.
**Ecosystem role:** web | **Status:** active

#### web-file
**What:** File API bindings.
**Key exports:** `File`, `Blob`, `FileReader`, `FileList`, `name`, `size`, `type_`, `readAsText`, `readAsArrayBuffer`
**When to use:** File upload handling, reading files from `<input type="file">`.
**Ecosystem role:** web | **Status:** active

#### canvas
**What:** HTML5 Canvas API bindings.
**Key exports:** `Context2D`, `getContext2D`, `fillRect`, `strokeRect`, `arc`, `beginPath`, `closePath`, `fill`, `stroke`, `setFillStyle`, `setStrokeStyle`, `drawImage`, `clearRect`
**When to use:** 2D graphics, drawing, custom visualizations.
**Note:** Also see `halogen-canvas` for Halogen integration.
**Ecosystem role:** web | **Status:** active

#### arraybuffer-types
**What:** ArrayBuffer and TypedArray type definitions.
**Key exports:** `ArrayBuffer`, `DataView`, `ArrayView`, `Int8Array`, `Uint8Array`, `Float32Array`, `Float64Array`, etc.
**When to use:** Binary data, WebGL, file processing, crypto. Foundation for typed array operations.
**Ecosystem role:** contrib | **Status:** active

### CLI & Configuration

#### optparse
**What:** Declarative command-line option parser (port of optparse-applicative).
**Key exports:** `Parser`, `ParserInfo`, `execParser`, `info`, `helper`, `strOption`, `option`, `flag`, `switch`, `argument`, `long`, `short`, `metavar`, `help`, `value`, `showDefault`, `subparser`, `command`
**When to use:** Building CLI tools with complex option parsing.
**Ecosystem role:** contrib | **Status:** active

#### argparse-basic
**What:** Simpler CLI argument parser.
**Key exports:** `ArgParser`, `argument`, `anyNotFlag`, `flag`, `flagHelp`, `flagInfo`, `fromRecord`, `parseArgs`, `printArgError`
**When to use:** Simpler CLI tools. Less boilerplate than `optparse`.
**Note:** By natefaubion.
**Ecosystem role:** community | **Status:** active

#### dotenv
**What:** Load environment variables from `.env` files.
**Key exports:** `loadFile`
**Ecosystem role:** community | **Status:** active

#### ansi
**What:** ANSI terminal escape codes.
**Key exports:** Terminal colors, cursor movement, screen clearing.
**Ecosystem role:** community | **Status:** active

#### dodo-printer
**What:** Pretty-printer library (Wadler-Lindig style).
**Key exports:** `Doc`, `text`, `lines`, `indent`, `group`, `break`, `flatAlt`, `print`
**When to use:** Generating formatted output (code generators, error messages).
**Note:** By natefaubion. Powers `tidy` (PureScript formatter).
**Ecosystem role:** community | **Status:** active

### URLs & HTTP Types

#### uri
**What:** URI parsing and representation.
**Key exports:** `URI`, `parse`, `print`, `Authority`, `Host`, `Port`, `Path`, `Query`, `Fragment`
**Ecosystem role:** contrib | **Status:** active

#### media-types
**What:** MIME/media type representation.
**Key exports:** `MediaType`, `applicationJSON`, `textHTML`, `textPlain`
**Ecosystem role:** contrib | **Status:** active

#### http-methods
**What:** HTTP method type.
**Key exports:** `Method(..)`, `GET`, `POST`, `PUT`, `DELETE`, `PATCH`
**Ecosystem role:** contrib | **Status:** active

#### form-urlencoded
**What:** URL-encoded form data.
**Key exports:** `encode`, `decode`
**Ecosystem role:** contrib | **Status:** active

### Database

#### postgresql
**What:** PostgreSQL client bindings.
**Key exports:** `Pool`, `Client`, `Query`, `newPool`, `query`, `execute`, `withClient`, `withTransaction`
**When to use:** PostgreSQL database access from Node.js.
**Ecosystem role:** community | **Status:** active

#### node-sqlite3
**What:** SQLite3 bindings for Node.js.
**Key exports:** `Database`, `open`, `close`, `run`, `get`, `all`, `prepare`
**When to use:** SQLite database access.
**Ecosystem role:** community | **Status:** active

### Other Important Packages

#### tidy
**What:** PureScript source code formatter.
**Key exports:** Command-line tool, not typically imported as a library.
**When to use:** Formatting PureScript code. Run via `purs-tidy`.
**Note:** By natefaubion.
**Ecosystem role:** community | **Status:** active

#### tidy-codegen
**What:** PureScript code generation library.
**Key exports:** Code generation combinators for producing PureScript source.
**When to use:** Writing tools that generate PureScript code.
**Note:** By natefaubion.
**Ecosystem role:** community | **Status:** active

#### convertable-options
**What:** Type-safe optional arguments via row types.
**Key exports:** `class ConvertOption`, `class ConvertOptionsWithDefaults`, `convertOptionsWithDefaults`
**When to use:** FFI bindings where JS functions take optional config objects.
**Note:** By natefaubion. Pattern: define defaults, let callers override any subset.
**Ecosystem role:** community | **Status:** active

#### graphs
**What:** Graph data structures.
**Key exports:** `Graph`, `topologicalSort`, `vertices`, `edges`
**Ecosystem role:** core | **Status:** active

#### unicode
**What:** Unicode character classification and case conversion.
**Key exports:** `isAlpha`, `isDigit`, `isAlphaNum`, `isUpper`, `isLower`, `isSpace`, `toUpper`, `toLower`, `GeneralCategory(..)`
**When to use:** Character classification that goes beyond ASCII.
**Ecosystem role:** contrib | **Status:** active

#### options
**What:** Applicative option parsing for record-style configuration.
**Key exports:** `Options`, `optional`, `required`, `defaulted`
**Ecosystem role:** contrib | **Status:** active

#### js-date
**What:** JavaScript `Date` object bindings.
**Key exports:** `JSDate`, `now`, `parse`, `fromTime`, `getTime`, `toISOString`, `toDateString`
**When to use:** Interop with JS Date. For pure date logic, use `datetime` instead.
**Ecosystem role:** contrib | **Status:** active

#### js-timers
**What:** JavaScript timer bindings (setTimeout, setInterval).
**Key exports:** `setTimeout`, `setInterval`, `clearTimeout`, `clearInterval`, `TimeoutId`, `IntervalId`
**When to use:** Low-level timers. In `Aff`, prefer `delay` from `Effect.Aff`.
**Ecosystem role:** contrib | **Status:** active

#### pipes
**What:** Streaming library (port of Haskell's pipes).
**Key exports:** `Producer`, `Consumer`, `Pipe`, `Effect`, `yield`, `await`, `runEffect`, `(>->)`
**When to use:** Composable streaming with back-pressure.
**Ecosystem role:** community | **Status:** active

#### coroutines
**What:** Coroutine-based streaming.
**Key exports:** `Co`, `Process`, `emit`, `consumer`, `producer`, `transform`, `connect`
**Ecosystem role:** contrib | **Status:** active

#### js-promise-aff
**What:** Convert between JS Promises and Aff.
**Key exports:** `toAff`, `toAffE`, `fromAff`
**When to use:** Bridging Promise-based JS APIs into `Aff`.
**Ecosystem role:** contrib | **Status:** active

#### js-promise
**What:** JS Promise bindings.
**Key exports:** `Promise`, `new`, `then_`, `catch_`, `resolve`, `reject`, `all`, `race`
**When to use:** Low-level Promise interop. Prefer `js-promise-aff` to lift into `Aff`.
**Ecosystem role:** contrib | **Status:** active

#### aff-promise
**What:** Alternative Promise-Aff bridge.
**Key exports:** `toAff`, `toAffE`, `Promise`
**Note:** By nwolverson. Similar to `js-promise-aff`. Both are widely used; check which the project already uses.
**Ecosystem role:** community | **Status:** active

#### uuid
**What:** UUID generation and parsing.
**Key exports:** `UUID`, `genUUID`, `parseUUID`, `toString`
**Ecosystem role:** community | **Status:** active

#### css
**What:** CSS types and generation.
**Key exports:** CSS property types, selectors, rendering.
**When to use:** Type-safe CSS generation.
**Ecosystem role:** contrib | **Status:** active

#### colors
**What:** Color types and manipulation.
**Key exports:** `Color`, `rgb`, `hsl`, `toHexString`, `lighten`, `darken`, `saturate`, `desaturate`, `complementary`
**Ecosystem role:** contrib | **Status:** active

#### mmorph
**What:** Monad morphisms — natural transformations between monads.
**Key exports:** `class MFunctor`, `hoist`, `class MMonad`, `embed`, `generalize`
**When to use:** Changing the base monad of a transformer stack. `hoist` lifts a natural transformation through a transformer.
**Ecosystem role:** community | **Status:** active

#### checked-exceptions
**What:** Tracked exceptions in the type system using row types.
**Key exports:** `class Throw`, `class Catch`, `throw`, `catch`, `safe`
**When to use:** When you want exceptions tracked in types without full `ExceptT`.
**Note:** By natefaubion.
**Ecosystem role:** community | **Status:** active

#### now
**What:** Get the current date/time in `Effect`.
**Key exports:** `now`, `nowDate`, `nowDateTime`
**Ecosystem role:** contrib | **Status:** active

#### const
**What:** The `Const` functor.
**Key exports:** `Const(..)` — a functor that ignores its second parameter. Useful for phantom type programming and as a building block for lenses.
**Ecosystem role:** core | **Status:** active

#### unfoldable
**What:** `Unfoldable` type class (dual of `Foldable`).
**Key exports:** `class Unfoldable`, `class Unfoldable1`, `unfoldr`, `replicate`, `replicateA`, `none`, `singleton`, `range`, `fromMaybe`, `toUnfoldable`
**When to use:** Converting between collection types. `toUnfoldable` from `Map`/`Set` produces any `Unfoldable`.
**Ecosystem role:** core | **Status:** active

#### identity
**What:** The `Identity` monad/functor.
**Key exports:** `Identity(..)` — trivial monad wrapper. Used as base monad for pure transformer stacks.
**Ecosystem role:** core | **Status:** active

---

## Ecosystem Context

### Key Organizations
- **purescript** — Core libraries maintained by the compiler team (prelude, effect, arrays, etc.)
- **purescript-contrib** — Community-maintained essential libraries (aff, argonaut, nullable, etc.)
- **purescript-node** — Node.js bindings (node-fs, node-buffer, node-streams, etc.)
- **purescript-web** — Web API bindings (web-dom, web-html, web-events, etc.)
- **purescript-halogen** — Halogen web framework
- **purescript-react** — React bindings (react-basic, react-basic-hooks, react-basic-dom)
- **rowtype-yoga** — yoga-json, yoga-fetch, and other "yoga" packages

### Key Authors
- **garyb** (Gary Burgess) — codec, codec-argonaut, codec-json, debug, indexed-monad, functor1. The codec approach to JSON is his design.
- **natefaubion** (Nate Faubion) — variant, run, routing-duplex, language-cst-parser, tidy, tidy-codegen, dodo-printer, heterogeneous, convertable-options, argparse-basic, checked-exceptions, call-by-name, node-workerbees, psa-utils. Prolific author of sophisticated type-level libraries.
- **thomashoneyman** (Thomas Honeyman) — halogen-hooks, Real World Halogen (tutorial app), PureScript documentation. Key educator in the community.
- **paf31** (Phil Freeman) — PureScript creator. Many core libraries.
- **hdgarrood** (Harry Garrood) — Pursuit (package documentation site), many core/contrib packages.
- **justinwoo** — simple-json, node-sqlite3, and many registry/tooling contributions.

### Build Tooling
- **spago** — Package manager and build tool. Config is `spago.yaml` (current) or `spago.dhall` (legacy). Subcommands: `build`, `test`, `run`, `install`, `bundle`, `docs`.
- **purs** — The compiler. Invoked via spago. `output/` directory contains compiled JS modules.
- **purs-tidy** — Code formatter. Run via `npx purs-tidy format-in-place "src/**/*.purs"`.
- **esbuild** / **webpack** — Bundle the `output/` JS for browser. PureScript compiles to ES modules.
- **Pursuit** — Package documentation site (pursuit.purescript.org). Like Haskell's Hackage/Haddock.

### Config File: `spago.yaml`
```yaml
package:
  name: my-package
  dependencies:
    - prelude
    - effect
    - aff
    - halogen
  test:
    main: Test.Main
    dependencies:
      - spec
      - spec-discovery
workspace:
  packageSet:
    registry: 71.0.0   # package set version
  extraPackages: {}     # additional/override packages
```

### Package Sets
PureScript uses curated package sets (like Haskell's Stackage). A package set is a consistent set of package versions known to compile together. The registry at `registry.purescript.org` publishes numbered sets (currently at 71.0.0). You pin to a set version in `spago.yaml`.

