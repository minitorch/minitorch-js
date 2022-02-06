enum Mode {
  Train,
  Eval,
}

class Module {
  mode: Mode;
  _modules: any;
  _parameters: any;

  constructor() {
    this._modules = {};
    this._parameters = {};
    this.mode = Mode.Train;
  }
}
