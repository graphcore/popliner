### Popart Bert training

Instructions on how to run this model can be found in:
https://github.com/graphcore/examples/tree/master/nlp/bert/pytorch

This report was generated with this particular command:
```cmd
CXX=clang++ POPART_LOG_LEVEL=WARN POPLAR_ENGINE_OPTIONS='{"autoReport.directory": "profile_demo_tiny_128", "autoReport.all": "true"}' POPART_POPLINER_OUTLINER_REGEX='(?:^|\/)(?:[L|l]ayer|blocks|encoder)[\/_\.]?(\d+)' python run_squad.py --config demo_tiny_128 --layers-per-ipu 3 --ipus-per-replica 1 --dataset generated --compile-only
```
