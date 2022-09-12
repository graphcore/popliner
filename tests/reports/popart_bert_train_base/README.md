### Popart Bert training

Instructions on how to run this model can be found in:
https://github.com/graphcore/examples/tree/master/nlp/bert/popart

This report was generated with this particular command:
```cmd
POPART_POPLINER_OUTLINER_REGEX='(?:^|\/)(?:[L|l]ayer|blocks|encoder)[\/_\.]?(\d+)' python3 bert.py --config configs/pretrain_base_128.json --layers-per-ipu 12 --replication-factor 1 --profile True --profile-dir profile_pretrain_base_128 --compile-only --generated-data --pipeline false --encoder-start-ipu 0
```
