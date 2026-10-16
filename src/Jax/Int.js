// int32 arrays and integer ops. jax-js keeps int32 arithmetic in int32
// (add, sub, mul wrap like i32 on every backend), so integer programs run
// exactly up to 2^31 instead of float32's 2^24.
import { numpy as np } from "@jax-js/jax";

export const arrayInt32Impl = (xs) => np.array(xs, { dtype: "int32" });
export const fullInt32Impl = (shape, v) => np.full(shape, v, { dtype: "int32" });
export const toInt32Impl = (a) => a.astype("int32");
export const toFloat32Impl = (a) => a.astype("float32");
export const dtypeImpl = (a) => a.dtype;

export const addScalarIntImpl = (a, n) => a.add(np.array(n, { dtype: "int32" }));
export const mulScalarIntImpl = (a, n) => a.mul(np.array(n, { dtype: "int32" }));
export const floorDivideImpl = (a, b) => np.floorDivide(a, b);
export const remainderImpl = (a, b) => np.remainder(a, b);
export const rightShiftImpl = (a, b) => np.rightShift(a, b);
export const leftShiftImpl = (a, b) => np.leftShift(a, b);
export const bitwiseAndImpl = (a, b) => np.bitwiseAnd(a, b);
export const bitwiseOrImpl = (a, b) => np.bitwiseOr(a, b);
export const bitwiseXorImpl = (a, b) => np.bitwiseXor(a, b);
export const negativeImpl = (a) => np.negative(a);
export const absoluteImpl = (a) => np.absolute(a);
export const maximumImpl = (a, b) => np.maximum(a, b);
export const minimumImpl = (a, b) => np.minimum(a, b);

// Comparisons give bool arrays; `where` selects.
export const greaterImpl = (a, b) => np.greater(a, b);
export const greaterEqualImpl = (a, b) => np.greaterEqual(a, b);
export const lessImpl = (a, b) => np.less(a, b);
export const equalImpl = (a, b) => np.equal(a, b);
export const whereImpl = (c, a, b) => np.where(c, a, b);
