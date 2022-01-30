var ops = require("ndarray-ops")
import * as nj from 'numjs';

var mat = nj.array([[1.0, 2.0], [3.0, 4.0]]);
var mat2 = nj.array([[1.0, 2.0], [3.0, 4.0]]);

function mul(x: nj.NdArray<any>, y: nj.NdArray<any>) {
  return nj.multiply(x, y)
}

function id(x: nj.NdArray<any>) {
  return x
}

function eq(x: nj.NdArray<any>, y: nj.NdArray<any>) {
  if (nj.equal(x, y)) {
    return nj.array([1.0])
  } else {
    return nj.array([0.0])
  }
}

function add(x: nj.NdArray<any>, y: nj.NdArray<any>) {
  return nj.add(x, y)
}

function max(x: nj.NdArray<any>, y: nj.NdArray<any>) {
}

function lt(x: nj.NdArray<any>, y: nj.NdArray<any>) {
}

function sigmoid(x: nj.NdArray<any>, y: nj.NdArray<any>) {
}

function relu(x: nj.NdArray<any>, y: nj.NdArray<any>) {
}

function inv(x: nj.NdArray<any>) {
}

function relu_back(x: nj.NdArray<any>, d: nj.NdArray<any>) {
}

function log_back(x: nj.NdArray<any>, d: nj.NdArray<any>) {
}
