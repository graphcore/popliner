### Popart Bert training

Instructions on how to run this model can be found in:
https://github.com/graphcore/examples/tree/master/applications/popart/bert

This report was generated with this particular command:
```cmd
POPLAR_ENGINE_OPTIONS='{"autoReport.outputDebugInfo":"true", "autoReport.outputGraphProfile":"true", "autoReport.outputLoweredVars":"true", "autoReport.directory":"./report_20220223_base","autoReport.outputExecutionProfile":"false","debug.allowOutOfMemory":"true"}' \
python3 ./bert.py --config configs/mk2/pretrain_base_128.json --synthetic-data --compile-only \
--encoder-start-ipu 0 --layers-per-ipu 12 --replication-factor 1 --replicated-tensor-sharding false --pipeline false
```
