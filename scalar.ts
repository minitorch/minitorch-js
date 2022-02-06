function central_difference(
  f: (x: number[]) => number,
  vals: number[],
  arg: number,
  eps: number
) {
  var vals2 = [...vals];
  vals2[arg] = vals2[arg] + eps / 2;
  vals[arg] = vals[arg] - eps / 2;
  console.log(vals2);
  console.log(vals);
  return (f(vals2) - f(vals)) / eps;
}

var tst = [3.0, 2.0];
var tst2 = [3.1, 2.0];

function fn_test(x: number[]): number {
  return x[0] * x[0] + 5 * x[1] + 3;
}

var foo = central_difference(fn_test, tst, 0, 0.1);
console.log(foo);
// d / dx x^2 (3) = 6.0
var foo = central_difference(fn_test, tst, 1, 0.1);
// d/ dx 5 * x (2) = 5.0
console.log(foo);

class Variable {
}

class Scalar extends Variable {
  data: number;
  history?: Hist;
  constructor(data: number, history: Hist) {
    super();
    this.data = data;
    this.history = history;
  }
}


abstract class FunctionBase {

}

abstract class ScalarFunction1 extends FunctionBase {
  abstract forward(x: number): number;

  apply(val: Scalar) {
    var need_grad = ! (history === undefined)
    var c = this.forward(val.data)
    var back = undefined
    if (need_grad) {
      back = new Hist(this,[val]);
    }
    return new Scalar(c, back);
  }
}

abstract class ScalarFunction2 {
  abstract forward(x: number, y: number): number;
  // abstract apply(x: Scalar, y: Scalar): Scalar;
}

class Mul extends ScalarFunction2 {
  x: number;
  y: number;
  constructor() {
    super();
    this.x = 0.0;
    this.y = 0.0;
  }
  forward(x: number, y: number): number {
    this.x = x;
    this.y = y;
    return x * y;
  }

  backward(d_out: number): number[] {
    var result = [this.y * d_out, this.x * d_out];
    return result;
  }
}

class Add extends ScalarFunction2 {
  forward(x: number, y: number): number {
    return x + y;
  }

  backward(d_out: number): number[] {
    return [1.0 * d_out, 1.0 * d_out];
  }
}

class Inv extends ScalarFunction1 {
  x: number;

  forward(x: number): number {
    this.x = x;
    return 1 / x;
  }

  backward(d_out: number): number {
    return -1.0 * ((1 / x) ^ 2) * d_out;
  }
}

class Neg extends ScalarFunction1 {
  forward(x: number): number {
    return -1.0 * x;
  }

  backward(d_out: number): number {
    return -1.0 * d_out;
  }
}

class Hist { // History is a reserved class name
  last_fn: FunctionBase;
  inputs: Variable[];

  constructor(last_fn: FunctionBase, inputs: Variable[]) {
    this.last_fn = last_fn
    this.inputs = inputs
  }

}


