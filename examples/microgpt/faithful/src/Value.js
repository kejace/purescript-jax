// Module-private monotonic counter. Each call returns the next int.
// Used as a unique identity for `Value`s so the topological-sort step
// can deduplicate via a Set Int (PureScript Refs aren't directly Eq,
// so we tag).
//
// Module-state-as-counter is the standard trick for this — equivalent
// to Python's `id(self)` but explicit. Threading a counter Ref through
// every op via ReaderT would be cleaner functionally, but adds noise
// for no behavioral benefit.

let _counter = 0;
export const nextUidImpl = () => {
  _counter += 1;
  return _counter;
};

