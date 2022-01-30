var ndarray = require("ndarray");
var ops = require("ndarray-ops");

// some test values
var d = ndarray(new Float32Array([1.0, 1.0, 1.0, 1.0]), [2, 2]);
var x = ndarray(new Float32Array([1.0, 0.0, 2.0, 3.0]), [2, 2]);
var x2 = ndarray(new Float32Array([1.0, 0.0, -2.0, 3.0]), [2, 2]);

// readability hack, signatures of ndarray ops is any
type Tensor = any;

function apply2(
  x: Tensor,
  y: Tensor,
  fn: (result: Tensor, x: Tensor, y: Tensor) => Tensor
): Tensor {
  var result = ndarray(new Float32Array(x.data), x.shape);
  fn(result, x, y);
  return result;
}

function apply(x: Tensor, fn: (result: Tensor, x: Tensor) => Tensor): Tensor {
  var result = ndarray(new Float32Array(x.data), x.shape);
  fn(result, x);
  return result;
}

function mul(x: Tensor, y: Tensor): Tensor {
  return apply2(x, y, ops.mul);
}

function id(x: Tensor): Tensor {
  return x;
}

function eq(x: Tensor, y: Tensor): Tensor {
  return apply2(x, y, ops.eq);
}

function add(x: Tensor, y: Tensor): Tensor {
  return apply2(x, y, ops.add);
}

function max(x: Tensor, y: Tensor): Tensor {
  return apply2(x, y, ops.max);
}

function lt(x: Tensor, y: Tensor): Tensor {
  return apply2(x, y, ops.lt);
}

function sigmoid(x: Tensor): Tensor {
  var result = ndarray(new Float32Array(x.data), x.shape);
  ops.assign(result, x);
  ops.mulseq(result, -1.0);
  ops.expeq(result);
  ops.addseq(result, 1.0);
  ops.recipeq(result);
  return result;
}

function relu(x: Tensor): Tensor {
  var result = ndarray(new Float32Array(x.data), x.shape);
  var comp = ndarray(new Float32Array(x.data), x.shape);
  ops.gts(comp, x, 0.0);
  ops.mul(result, x, comp);
  return result;
}

function inv(x: Tensor): Tensor {
  return apply(x, ops.recip);
}

function inv_back(x: Tensor, d: Tensor): Tensor {
  var result = ndarray(new Float32Array(x.data), x.shape);
  ops.assign(result, x);
  ops.muleq(result, result);
  ops.mulseq(result, -1.0);
  ops.muleq(result, d);
  return result;
}

function relu_back(x: Tensor, d: Tensor): Tensor {
  var result = ndarray(new Float32Array(x.data), x.shape);
  ops.gts(result, x, 0.0);
  ops.muleq(result, d);
  return result;
}

function log_back(x: Tensor, d: Tensor): Tensor {
  var result = ndarray(new Float32Array(x.data), x.shape);
  ops.assign(result, x);
  ops.recipeq(result);
  ops.muleq(result, d);
  return result;
}

function map(fn: (x: Tensor) => Tensor) {
  // TODO
}
