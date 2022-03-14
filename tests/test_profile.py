# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from popliner import operation_list
from pva import openReport  # pylint: disable=no-name-in-module
from popliner.greedy_solver import GreedySolver
import pytest


@pytest.fixture(scope='module')
def efficientnet():
    report = openReport('tests/efficientnet/profile.pop',
                        'tests/efficientnet/debug.cbor')
    test_operations = operation_list.OperationList(report)

    yield (report, test_operations)


# NOTE: This will fail unless profile files are in tests/efficientnet/
def test_values(efficientnet):
    (_, operations) = efficientnet

    assert(sum([item.operation.vertex_state_bytes() for item in operations]) == 1882988)
    assert(sum([len(item.operation.programs) for item in operations]) == 1282)
    assert(sum([item.operation.vertex_code_bytes() for item in operations]) == 40855000)
    assert(sum([item.operation.exchange_code() for item in operations]) == 1112688)
    assert(sum([
        item.operation.variable_bytes() for item in operations]) == 617314032)

    assert(len(operations) == 237)


def test_json_output(efficientnet):

    (report, operations) = efficientnet
    solver = GreedySolver(report, operations)
    assert(solver.solve(4, 638976))
    assert(solver.get_splits_as_json() == """\
[
    {
        "layer_from": "blocks_0",
        "layer_to": "POSTAMBLE",
        "mem": {
            "total_mem": 182285350,
            "variables": 170958840,
            "vertex_code": 5164382,
            "vertex_state": 1497068,
            "exchange_code": 1081496,
            "control_code": 3583564
        }
    }
]""")


def test_operation_breakdown(efficientnet):

    (_, operations) = efficientnet
    assert(operations.as_csv(",", 27) == """\
Layer,Full name,Vertex state,Vertex code,Exchange code,Control code,Variables,
blocks_0(deduced),LoopCond,20,384,0,28,0
blocks_0(deduced),LoopCond,0,0,0,9972,0

blocks_0(deduced),optimized/efficientnet-edgetpu-L/model/stem/tpu_batch_normalization/batchnorm/mul_1,46944,1343200,7748,13052,0
blocks_0(deduced),optimized/efficientnet-edgetpu-L/model/stem/tpu_batch_normalization/batchnorm/add_1,0,0,528,300,0
blocks_0(deduced),optimized/efficientnet-edgetpu-L/model/stem/Relu,640,0,0,256,0
blocks_0,optimized/efficientnet-edgetpu-L/model/blocks_0/tpu_batch_normalization/batchnorm/mul_1,6144,61254,5084,10424,4967936
blocks_0,optimized/efficientnet-edgetpu-L/model/blocks_0/tpu_batch_normalization/batchnorm/add_1,0,0,4420,1568,13096064
blocks_0,optimized/efficientnet-edgetpu-L/model/blocks_0/Relu,0,0,0,68,12845056
blocks_0,optimized/efficientnet-edgetpu-L/model/blocks_0/tpu_batch_normalization_1/batchnorm/mul_1,37888,449436,1124,72152,12930240
blocks_0,optimized/efficientnet-edgetpu-L/model/blocks_0/tpu_batch_normalization_1/batchnorm/add_1,8,246,0,4288,6701456
blocks_1,optimized/efficientnet-edgetpu-L/model/blocks_1/tpu_batch_normalization/batchnorm/mul_1,257920,0,220532,71408,4128271
blocks_1,optimized/efficientnet-edgetpu-L/model/blocks_1/tpu_batch_normalization/batchnorm/add_1,247396,0,0,23608,13177056
blocks_1,optimized/efficientnet-edgetpu-L/model/blocks_1/Relu,80,0,0,40,12845056
blocks_1,optimized/efficientnet-edgetpu-L/model/blocks_1/tpu_batch_normalization_1/batchnorm/mul_1,8892,362066,1200,46600,12913856
blocks_1,optimized/efficientnet-edgetpu-L/model/blocks_1/tpu_batch_normalization_1/batchnorm/add_1,0,0,501796,6144,6701456
blocks_2,optimized/efficientnet-edgetpu-L/model/blocks_2/tpu_batch_normalization/batchnorm/mul_1,208128,3308560,245152,67712,4275727
blocks_2,optimized/efficientnet-edgetpu-L/model/blocks_2/tpu_batch_normalization/batchnorm/add_1,16704,7782,40164,16768,26129024
blocks_2,optimized/efficientnet-edgetpu-L/model/blocks_2/Relu,0,0,0,24,25690112
blocks_2,optimized/efficientnet-edgetpu-L/model/blocks_2/tpu_batch_normalization_1/batchnorm/mul_1,8736,360354,0,33024,25780464
blocks_2,optimized/efficientnet-edgetpu-L/model/blocks_2/tpu_batch_normalization_1/batchnorm/add_1,0,0,0,5496,1066356
blocks_3,optimized/efficientnet-edgetpu-L/model/blocks_3/tpu_batch_normalization/batchnorm/mul_1,8744,360600,1064,61416,2832320
blocks_3,optimized/efficientnet-edgetpu-L/model/blocks_3/tpu_batch_normalization/batchnorm/add_1,0,0,0,7728,8200680
blocks_3,optimized/efficientnet-edgetpu-L/model/blocks_3/Relu,8736,360354,0,11648,8028160
blocks_3,optimized/efficientnet-edgetpu-L/model/blocks_3/tpu_batch_normalization_1/batchnorm/mul_1,8936,360600,24,63072,8183075
blocks_3,optimized/efficientnet-edgetpu-L/model/blocks_3/ArithmeticOptimizer/AddOpsRewrite_Leaf_1_Add,8832,360354,0,17664,3010560
blocks_3,optimized/efficientnet-edgetpu-L/model/blocks_3/ArithmeticOptimizer/AddOpsRewrite_Add,0,0,0,11264,1066356
""")


def test_layer_breakdown(efficientnet):

    (_, operations) = efficientnet
    assert(operations.layers_as_csv(",") == """\
Layer,Vertex state,Vertex code,Exchange code,Control code,Variables,
blocks_0,91644,1853864,18904,112108,95266768
blocks_1,514288,362066,723528,147800,77103343
blocks_2,233568,3668914,285316,123024,127164959
blocks_3,35248,360600,1088,172792,84203435
blocks_4,35144,360600,2344,171688,84203435
blocks_5,17480,360600,1832,108864,82893176
blocks_6,43872,360354,4396,189656,43128852
blocks_7,26312,360600,2772,152208,43128852
blocks_8,35136,360354,2812,168376,43128852
blocks_9,26304,360354,4732,170784,43128852
blocks_10,26312,360600,2068,154000,43138456
blocks_11,35040,360354,3096,153904,19101206
blocks_12,52624,360600,4060,242528,21126601
blocks_13,43792,360600,3224,212064,21136097
blocks_14,52616,360600,5048,241920,21136097
blocks_15,43888,360600,2876,218656,21136184
blocks_16,43792,360600,4144,222256,21126601
blocks_17,52616,360600,4184,238552,21126601
blocks_18,43888,360600,2048,194704,27016912
blocks_19,43784,360600,4344,249440,45350763
blocks_20,52624,360600,3448,242112,45355111
blocks_21,43792,360600,2676,221496,45350763
blocks_22,52616,360600,4516,251352,45355111
blocks_23,43888,360600,3596,228600,45345430
blocks_24,52520,360600,4308,249600,40121926
blocks_25,52528,360600,2336,238904,24663140
blocks_26,35144,360600,4768,223576,24663172
POSTAMBLE,52528,360600,4224,257048,0
""")
