### Popart Bert inference

Instructions on how to run this model can be found in:
https://github.com/graphcore/examples/tree/master/nlp/bert/popart

This report was generated with this particular command:
```cmd
POPART_POPLINER_OUTLINER_REGEX='(?:^|\/)(?:[L|l]ayer|blocks|encoder)[\/_\.]?(\d+)' python3 bert.py --config configs/squad_large_128_inf.json --profile True --profile-dir profile_squad_large_128_inf --compile-only --generated-data
```
