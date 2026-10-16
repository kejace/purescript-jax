// Helpers for Test.Exact: reference values computed in plain JS.
export const collatzLengths = (xs) => xs.map((x) => { let n = x, s = 0; while (n !== 1) { n = n % 2 === 0 ? n / 2 : 3 * n + 1; s++; } return s; });
