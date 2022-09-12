# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from popliner import operation_list
from pva import openReport
from popliner.greedy_solver import GreedySolver
from popliner.stage import Stage
import popliner.parse_args
import numpy as np
import json
import pytest


@pytest.fixture(scope='module')
def popart_bert_inf(tmpdir_factory):
    pickle_path = tmpdir_factory.mktemp("popart_bert_inf").join("test.pickle")
    report = openReport('tests/popart-bert-inf/profile.pop',
                        'tests/popart-bert-inf/debug.cbor')

    test_operations = operation_list.OperationList(report)

    operation_list.OperationList(
        report, popliner.parse_args.get_args(['--save-to-file', str(pickle_path)]))
    test_operations2 = operation_list.OperationList.from_file(pickle_path)

    yield [test_operations, test_operations2]


# NOTE: This will fail unless profile files are in tests/popart-bert-inf/
def test_values(popart_bert_inf):
    for operations in popart_bert_inf:
        stage = Stage()
        for op in operations:
            stage.add(op)

        assert(np.sum(stage.vertex_state_bytes_by_tile()) == 25_460_686)
        assert(np.sum(len(stage.program_ids)) == 3_105)
        assert(np.sum(stage.code_bytes_by_tile()) == 9_887_022)
        assert(np.sum(stage.exchange_code_by_tile()) == 27_805_164)
        assert(np.sum(stage.max_vars_usage()) == 718_514_294)

        assert(len(operations) == 450)


def test_json_output(popart_bert_inf):
    for operations in popart_bert_inf:
        solver = GreedySolver(operations)
        assert(solver.solve(2, 350000))
        assert(json.dumps(solver.get_splits_totals(), indent=4) == """\
[
    {
        "layer_from": null,
        "layer_to": "11",
        "mem": {
            "total_mem": 470821068,
            "max_tile_mem": 336670,
            "variables": 416249974,
            "vertex_code": 9664672,
            "vertex_state": 12826418,
            "exchange_code": 15544264,
            "control_code": 16535740
        }
    },
    {
        "layer_from": "12",
        "layer_to": "23",
        "mem": {
            "total_mem": 403195814,
            "max_tile_mem": 291170,
            "variables": 353891960,
            "vertex_code": 8081590,
            "vertex_state": 12634268,
            "exchange_code": 12260900,
            "control_code": 16327096
        }
    }
]""")


def test_operation_breakdown(popart_bert_inf):
    for operations in popart_bert_inf:
        assert operations.as_csv(",", 27) == """\
Layer,Note,Full name,Vertex state,Vertex code,Exchange code,Control code,Variables,
None,0_input_layers,Anonymous,0,0,20604,17680,0
None,tensor_filter,Tensor/segments,0,0,0,252,0
None,tensor_filter,Tensor/seq_pad_idx,8,688,0,24,0

0,,Layer0/Attention/Z/Mask/Less,64,1055,64,136,516
0,,Layer0/Attention/Z/Mask/Cast,48,2160,0,80,384
0,,Layer0/Attention/Z/Mask/Sub,64,1048,0,80,520
0,,Layer0/Attention/Z/Mask/Mul,48,1044,0,80,264

None,0_input_layers,Embedding/OneHot,6784,240139,5612,15452,0
None,0_input_layers,Embedding/MatMul,59504,1670712,40304,50044,0
None,0_input_layers,Embedding/Gather,40960,1090560,90864,31136,0

0,inputs_regex_layer_0,Embedding/Gather,92496,1258490,543924,82320,62259822

None,0_input_layers,Embedding/Add,15360,337156,209040,31488,0
None,0_input_layers,Embedding/Add,16416,351012,87776,39136,0
None,deduced_from_following_operation,Call(1254),221312,2991825,227504,267564,0
None,0_input_layers,Embedding/GroupNormalization,221120,2973137,227504,261016,0

0,,Layer0/Attention/MatMul,86272,2372928,520300,104228,6638784
0,,Layer0/Attention/Z/MatMul,94208,2137088,343312,115008,605312
0,,Layer0/Attention/Z/Mul,15360,367908,0,25600,526848
0,,Layer0/Attention/Z/ApplyMask,20480,207360,28096,30720,559872
0,,Layer0/Attention/Z/Softmax,160768,2295839,102048,201216,582912
0,,Layer0/Attention/Z/MatMul,65536,1794048,198508,82800,858240
0,,Layer0/Attention/MatMul,86272,2372928,472528,105216,2444480
0,,Layer0/Attention/Add,17472,380228,42468,34880,783616
0,deduced_from_input_names,Call(1255),221216,2982481,269544,273140,574232
0,,Layer0/Attention/GroupNormalization,221120,2973137,227504,261016,0
0,,Layer0/FF/1/MatMul,86272,2372928,443848,103636,8735936
0,,Layer0/FF/1/Add,25088,235872,39380,35840,1125032"""


def test_layer_breakdown(popart_bert_inf):
    for operations in popart_bert_inf:
        assert(operations.layers_as_csv(",") == """\
Name,Vertex state,Vertex code,Exchange code,Control code,Variables,Total (MB),
None,360728,6387468,681980,453540,0,8
0,1136326,8983097,3657184,1438688,139178102,147
1,1050044,7876760,1280820,1360492,76814456,84
2,1050044,7876760,1023944,1354392,76814456,84
3,1050044,7876760,1018172,1354392,76814456,84
4,1050044,7876760,1019844,1354392,76814456,84
5,1050044,7876760,1017864,1354392,76814456,84
6,1050044,7876760,1016684,1354392,76814456,84
7,1050044,7876760,1014080,1354392,76814456,84
8,1050044,7876176,1010824,1354684,76814456,84
9,1050044,7876760,1006336,1354368,76814456,84
10,1050044,7876176,1009124,1354396,76814456,84
11,1050044,7876760,1014912,1354392,76814456,84
12,1050044,7876760,1014712,1354392,76814456,84
13,1050044,7876760,1014320,1354392,76814456,84
14,1050044,7876760,1017380,1354392,76814456,84
15,1050044,7876760,1017392,1354392,76814456,84
16,1050044,7876760,1012960,1354368,76814456,84
17,1050044,7876760,1015100,1354376,76814456,84
18,1050044,7876760,1017048,1354440,76814456,84
19,1049600,7869544,1006668,1361160,76814456,84
20,1050044,7876022,1016864,1354392,76814456,84
21,1049600,7869544,1011884,1361160,76814456,84
22,1049600,7869544,1009188,1361160,76814456,84
23,1085116,7876760,1107384,1408472,76836728,84
""")
