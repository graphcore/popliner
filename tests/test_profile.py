# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from popliner import operation_list
import numpy as np
from popliner.stage import Stage
from pva import openReport
from popliner.greedy_solver import GreedySolver
import json
import pytest


@pytest.fixture(scope='module')
def efficientnet():
    report = openReport('tests/efficientnet/profile.pop',
                        'tests/efficientnet/debug.cbor')
    test_operations = operation_list.OperationList(report)

    yield test_operations


@pytest.fixture(scope='module')
def poptorch_demo_tiny_128():
    report = openReport('tests/reports/poptorch_demo_tiny_128/profile.pop',
                        'tests/reports/poptorch_demo_tiny_128/debug.cbor')
    test_operations = operation_list.OperationList(report)

    yield test_operations


# NOTE: This will fail unless profile files are in tests/efficientnet/
def test_efficientnet_values(efficientnet):
    operations = efficientnet

    stage = Stage()
    for op in operations:
        stage.add(op)

    assert(np.sum(stage.vertex_state_bytes_by_tile()) == 18_579_720)
    assert(np.sum(len(stage.program_ids)) == 832)
    assert(np.sum(stage.code_bytes_by_tile()) == 17_381_396)
    assert(np.sum(stage.exchange_code_by_tile()) == 43_089_872)
    assert(np.sum(stage.max_vars_usage()) == 170_958_840)

    assert(len(operations) == 237)


def test_efficientnet_json_output(efficientnet):

    operations = efficientnet
    solver = GreedySolver(operations)
    assert(solver.solve(4, 638976))
    assert(json.dumps(solver.get_splits_totals(), indent=4) == """\
[
    {
        "layer_from": null,
        "layer_to": "26",
        "mem": {
            "total_mem": 258871736,
            "max_tile_mem": 194798,
            "variables": 170958840,
            "vertex_code": 17381396,
            "vertex_state": 18579720,
            "exchange_code": 43089872,
            "control_code": 8861908
        }
    }
]""")


def test_efficientnet_operation_breakdown(efficientnet):

    operations = efficientnet
    assert(operations.as_csv(",", 27) == """\
Layer,Note,Full name,Vertex state,Vertex code,Exchange code,Control code,Variables,
None,0_input_layers,LoopCond,20,384,0,28,0
None,0_input_layers,LoopCond,0,0,0,6540,0
None,0_input_layers,optimized/efficientnet-edgetpu-L/model/stem/tpu_batch_normalization/batchnorm/mul_1,1237236,2712768,1362140,134568,0
None,0_input_layers,optimized/efficientnet-edgetpu-L/model/stem/tpu_batch_normalization/batchnorm/add_1,87600,970559,23360,40880,0
None,0_input_layers,optimized/efficientnet-edgetpu-L/model/stem/Relu,11680,288787,0,11680,0

0,,optimized/efficientnet-edgetpu-L/model/blocks_0/tpu_batch_normalization/batchnorm/mul_1,76550,3308816,1658612,23608,4967936
0,,optimized/efficientnet-edgetpu-L/model/blocks_0/tpu_batch_normalization/batchnorm/add_1,313344,1233284,23564,133936,13096064
0,,optimized/efficientnet-edgetpu-L/model/blocks_0/Relu,70656,253184,0,23552,12845056
0,,optimized/efficientnet-edgetpu-L/model/blocks_0/tpu_batch_normalization_1/batchnorm/mul_1,70080,2029489,3439280,41280,12930240
0,,optimized/efficientnet-edgetpu-L/model/blocks_0/tpu_batch_normalization_1/batchnorm/add_1,87360,966519,0,34944,6701456

1,,optimized/efficientnet-edgetpu-L/model/blocks_1/tpu_batch_normalization/batchnorm/mul_1,100120,3562168,1218204,41240,4128271
1,,optimized/efficientnet-edgetpu-L/model/blocks_1/tpu_batch_normalization/batchnorm/add_1,287744,1233284,23596,128768,13177056
1,,optimized/efficientnet-edgetpu-L/model/blocks_1/Relu,70656,253184,0,23552,12845056
1,,optimized/efficientnet-edgetpu-L/model/blocks_1/tpu_batch_normalization_1/batchnorm/mul_1,70080,2029489,3536824,47168,12913856
1,,optimized/efficientnet-edgetpu-L/model/blocks_1/tpu_batch_normalization_1/batchnorm/add_1,87360,966519,0,34944,6701456

2,,optimized/efficientnet-edgetpu-L/model/blocks_2/tpu_batch_normalization/batchnorm/mul_1,76568,3308984,1830056,23576,4275727
2,,optimized/efficientnet-edgetpu-L/model/blocks_2/tpu_batch_normalization/batchnorm/add_1,326144,1233284,23640,139008,26129024
2,,optimized/efficientnet-edgetpu-L/model/blocks_2/Relu,70656,253184,0,23552,25690112
2,,optimized/efficientnet-edgetpu-L/model/blocks_2/tpu_batch_normalization_1/batchnorm/mul_1,65856,1904336,1727008,39216,25780464
2,,optimized/efficientnet-edgetpu-L/model/blocks_2/tpu_batch_normalization_1/batchnorm/add_1,21952,299514,27968,16464,1066356

3,,optimized/efficientnet-edgetpu-L/model/blocks_3/tpu_batch_normalization/batchnorm/mul_1,237480,4108696,1473688,86976,2832320
3,,optimized/efficientnet-edgetpu-L/model/blocks_3/tpu_batch_normalization/batchnorm/add_1,84000,913666,22508,39312,8200680
3,,optimized/efficientnet-edgetpu-L/model/blocks_3/Relu,11200,276673,0,11200,8028160
3,,optimized/efficientnet-edgetpu-L/model/blocks_3/tpu_batch_normalization_1/batchnorm/mul_1,71542,2049840,3453224,60068,8183075
3,,optimized/efficientnet-edgetpu-L/model/blocks_3/ArithmeticOptimizer/AddOpsRewrite_Leaf_1_Add,16464,304585,1267644,16896,3010560
3,,optimized/efficientnet-edgetpu-L/model/blocks_3/ArithmeticOptimizer/AddOpsRewrite_Add,21952,299514,26268,16464,1066356

4,,optimized/efficientnet-edgetpu-L/model/blocks_4/tpu_batch_normalization/batchnorm/mul_1,237480,4108696,1473688,86976,2832320\
""")


def test_efficientnet_layer_breakdown(efficientnet):

    operations = efficientnet
    assert(operations.layers_as_csv(",") == """\
Name,Vertex state,Vertex code,Exchange code,Control code,Variables,Total (MB),
None,1739814,7476105,2146852,505348,0,11
0,617990,7167086,5121456,257320,95266768,103
1,615960,7169582,4778624,275672,77103343,86
2,561176,6998878,3608672,241816,127164959,132
3,442638,7401152,6243332,230916,84203435,94
4,442638,7401152,4975652,230916,84203435,93
5,420488,7266901,2410944,192592,82893176,89
6,410756,7301845,3291668,252880,43128852,52
7,410756,7301845,2801844,252880,43128852,51
8,410756,7301845,2803072,252880,43128852,51
9,410756,7301845,2803024,252880,43128852,51
10,427220,7002331,3552140,280320,43138456,52
11,637678,6925325,1667228,436612,19101206,27
12,912724,7409879,1448348,527244,21126601,30
13,911152,7409787,1443544,526460,21136097,30
14,911152,7409787,1443232,526460,21136097,30
15,909584,7391686,1444204,526460,21136184,30
16,912720,7409787,1443504,527244,21126601,30
17,912724,7409879,1444772,527244,21126601,30
18,902176,7413442,1681352,518956,27016912,36
19,2135658,10356249,10521100,771984,45350763,66
20,2130954,10348869,10488540,769920,45355111,66
21,2135658,10356249,10488100,771984,45350763,66
22,2130954,10348869,10487852,769920,45355111,66
23,2138250,10370066,10488340,774108,45345430,66
24,875238,7007660,1783928,465088,40121926,48
25,3501396,8802284,3165828,666328,24663140,39
26,3501400,8802445,3165932,666328,24663172,39
""")


# NOTE: This will fail unless profile files are in tests/reports/poptorch_demo_tiny_128/
def test_poptorch_demo_tiny_128_values(poptorch_demo_tiny_128):
    operations = poptorch_demo_tiny_128

    stage = Stage()
    for op in operations:
        stage.add(op)

    assert(np.sum(stage.vertex_state_bytes_by_tile()) == 11_348_758)
    assert(np.sum(len(stage.program_ids)) == 4664)
    assert(np.sum(stage.code_bytes_by_tile()) == 24_414_353)
    assert(np.sum(stage.exchange_code_by_tile()) == 9_779_604)
    assert(np.sum(stage.max_vars_usage()) == 40_434_347)

    assert(len(operations) == 1025)


def test_poptorch_demo_tiny_128_json_output(poptorch_demo_tiny_128):

    operations = poptorch_demo_tiny_128
    solver = GreedySolver(operations)
    assert(solver.solve(4, 638976))
    assert(json.dumps(solver.get_splits_totals(), indent=4) == """\
[
    {
        "layer_from": null,
        "layer_to": "2",
        "mem": {
            "total_mem": 103077866,
            "max_tile_mem": 106267,
            "variables": 40434347,
            "vertex_code": 24414353,
            "vertex_state": 11348758,
            "exchange_code": 9779604,
            "control_code": 17100804
        }
    }
]""")


def test_poptorch_demo_tiny_128_operation_breakdown(poptorch_demo_tiny_128):

    operations = poptorch_demo_tiny_128
    assert(operations.as_csv(",", 27) == """\
Layer,Note,Full name,Vertex state,Vertex code,Exchange code,Control code,Variables,
None,0_input_layers,Anonymous,0,0,20596,17680,0
None,tensor_filter,Tensor/bert.encoder.layer.2.attention.self.value.bias,0,0,0,17664,0
None,tensor_filter,Tensor/adamGradientScaling___specific___bert.encoder.layer.1.attention.self.value.bias,0,0,0,2064,0
None,tensor_filter,Tensor/lossScaling_FLOAT16,8,688,0,28,0
None,tensor_filter,Tensor/randomSeed___fromHost,53016,276840,29712,164924,0
None,tensor_filter,Tensor/input/4,0,0,0,504,0
None,0_input_layers,Div,16,297,0,20,0
None,0_input_layers,Anonymous(419),24,292,0,48,0
None,0_input_layers,Anonymous(426),29508,277852,20608,100064,0
None,0_input_layers,Max,16,76,0,20,0
None,0_input_layers,Min,16,76,0,20,0
None,0_input_layers,Max,16,76,0,20,0
None,0_input_layers,Min,16,76,0,20,0
None,0_input_layers,Anonymous(425),29508,277852,12,94344,0
None,0_input_layers,Anonymous(424),29508,277852,12,94344,0
None,0_input_layers,Anonymous(423),29508,277852,12,94344,0
None,0_input_layers,Anonymous(422),29508,277852,12,94344,0
None,0_input_layers,Anonymous(421),29508,277852,12,94344,0
None,0_input_layers,Anonymous(420),29492,277852,12,100148,0
None,0_input_layers,bert/Cast,160,6057,0,272,0
None,0_input_layers,bert/Sub,480,6756,172,676,0
None,0_input_layers,bert/Mul,144,1524,0,240,0
None,0_input_layers,bert/embeddings/token_type_embeddings/Gather,39424,916608,28428,34452,0

0,inputs_regex_layer_0,bert/embeddings/Add,14784,247615,32504,30776,147456

None,0_input_layers,bert/embeddings/position_embeddings/Gather,91324,1088998,382568,86412,0
None,0_input_layers,bert/embeddings/Add,16512,277453,37144,36360,0
None,0_input_layers,bert/embeddings/LayerNorm/Groupnormalization,209300,2053191,142260,220532,0\
""")


def test_poptorch_demo_tiny_128_layer_breakdown(poptorch_demo_tiny_128):

    operations = poptorch_demo_tiny_128
    assert(operations.layers_as_csv(",") == """\
Name,Vertex state,Vertex code,Exchange code,Control code,Variables,Total (MB),
None,1693012,15896739,2159956,2872432,0,22
0,3810654,20595016,3602360,5472868,40020637,70
1,3259946,18292815,1736352,5083380,9603657,36
2,3289234,18266663,2625736,5136444,9600665,37
""")
