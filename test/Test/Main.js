// FFI for tiny test helpers used by Test.Main.

export const length = (xs) => xs.length;

export const allCloseImpl = (tol) => (a) => (b) => {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > tol) return false;
  }
  return true;
};
