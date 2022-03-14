# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from popliner import operation_list
from pva import openReport  # pylint: disable=no-name-in-module
from popliner.greedy_solver import GreedySolver
import pytest


@pytest.fixture(scope='module')
def popart_bert_inf():
    report = openReport('tests/popart-bert-inf/profile.pop',
                        'tests/popart-bert-inf/debug.cbor')
    test_operations = operation_list.OperationList(report)

    yield (report, test_operations)


# NOTE: This will fail unless profile files are in tests/popart-bert-inf/
def test_values(popart_bert_inf):
    (_, operations) = popart_bert_inf

    assert(sum([item.operation.vertex_state_bytes() for item in operations]) == 81222440)
    assert(sum([len(item.operation.programs) for item in operations]) == 8809)
    assert(sum([item.operation.vertex_code_bytes() for item in operations]) == 962623926)
    assert(sum([item.operation.exchange_code() for item in operations]) == 154322980)
    assert(sum([
        item.operation.variable_bytes() for item in operations]) == 6749866050)

    assert(len(operations) == 326)


def test_json_output(popart_bert_inf):

    (report, operations) = popart_bert_inf
    solver = GreedySolver(report, operations)
    assert(solver.solve(2, 350000))
    assert(solver.get_splits_as_json() == """\
[
    {
        "layer_from": "Layer0",
        "layer_to": "Layer11",
        "mem": {
            "total_mem": 381009389,
            "variables": 359204909,
            "vertex_code": 12065296,
            "vertex_state": 2358788,
            "exchange_code": 4188184,
            "control_code": 3192212
        }
    },
    {
        "layer_from": "Layer12",
        "layer_to": "Layer23",
        "mem": {
            "total_mem": 375421819,
            "variables": 359200285,
            "vertex_code": 8576682,
            "vertex_state": 1624732,
            "exchange_code": 3125692,
            "control_code": 2894428
        }
    }
]""")


def test_operation_breakdown(popart_bert_inf):

    (_, operations) = popart_bert_inf
    assert(operations.as_csv(",", 27) == """\
Layer,Full name,Vertex state,Vertex code,Exchange code,Control code,Variables,
Layer0,Layer0/Attention/Z/Mask/Less,64,1078,64,88,516
Layer0,Layer0/Attention/Z/Mask/Cast,48,2448,0,32,384
Layer0,Layer0/Attention/Z/Mask/Sub,64,1048,0,32,520
Layer0,Layer0/Attention/Z/Mask/Mul,48,1044,0,32,264

POSTAMBLE(keyword filter),Embedding/OneHot,6784,241250,5612,10980,0
POSTAMBLE(keyword filter),Embedding/MatMul,100712,2200496,77144,54036,0
POSTAMBLE(keyword filter),Embedding/Gather,186368,890880,97712,41984,0
POSTAMBLE(keyword filter),Embedding/Gather,362320,1028878,551968,81624,0
POSTAMBLE(keyword filter),Embedding/Add,16896,364292,210208,16896,0
POSTAMBLE(keyword filter),Embedding/Add,16512,353668,75108,22272,0

Layer0(deduced),Call,388928,3070632,208000,214004,280766

POSTAMBLE(keyword filter),Embedding/GroupNormalization,388736,3051944,208000,208148,0

Layer0(deduced),Call,95008,3052958,457736,82152,12897472

Layer0,Layer0/Attention/MatMul,86272,2203392,457736,63744,43635392

Layer0(deduced),Call,290846,4667442,491884,256516,4572681

Layer0,Layer0/Attention/Z/Mul,94230,1964998,344056,74524,4554759

Layer0(deduced),Call,853254,7156170,2142908,681976,75765566

Layer0,Layer0/Attention/GroupNormalization,388736,3051944,208000,208148,0
Layer0,Layer0/FF/1/MatMul,65536,1615872,196816,52464,4419072
Layer0,Layer0/FF/1/MatMul,108230,2285896,538764,72220,55375040
Layer0,Layer0/FF/2/MatMul,388736,3051944,208768,219252,892950
Layer0,Layer0/FF/2/MatMul,87040,2225152,422980,64256,54316800
Layer0,Layer0/FF/2/MatMul,86272,2203392,480680,61888,23907520
Layer0,Layer0/FF/2/Add,388736,3051944,208624,219252,892950
Layer0,Layer0/FF/GroupNormalization,388736,3051944,208000,208148,0

Layer1(deduced),Call,95008,3052958,457736,82152,12897472
""")


def test_layer_breakdown(popart_bert_inf):

    (_, operations) = popart_bert_inf
    assert(operations.layers_as_csv(",") == """\
Layer,Vertex state,Vertex code,Exchange code,Control code,Variables,
Layer0,1239524,8581256,3092592,1026636,82178071
POSTAMBLE,1113592,7740712,1270800,477348,0
Layer1,1238724,8576682,3094304,1041684,82173447
Layer2,1238724,8576682,3094704,1041684,82173447
Layer3,1244652,8576682,3094320,1050228,82173447
Layer4,1238724,8576682,3094340,1041940,82173447
Layer5,1238724,8576682,3094212,1041684,82173447
Layer6,1238724,8576682,3094052,1041684,82173447
Layer7,1238724,8576682,3093860,1041684,82173447
Layer8,1238724,8576682,3093916,1041684,82173447
Layer9,1238724,8576682,3094068,1041684,82173447
Layer10,1238724,8576682,3094136,1041684,82173447
Layer11,1241676,8576682,3094344,1045620,82173447
Layer12,1238724,8576682,3094628,1041684,82173447
Layer13,1238724,8576682,3094472,1042244,82173447
Layer14,1238724,8576682,3094656,1041940,82173447
Layer15,1238724,8576682,3094180,1041684,82173447
Layer16,1238724,8576682,3094060,1041684,82173447
Layer17,1238724,8576682,3093828,1041684,82173447
Layer18,1238724,8576682,3093932,1041684,82173447
Layer19,1238724,8576682,3093704,1041684,82173447
Layer20,1238724,8576682,3094168,1041684,82173447
Layer21,1238724,8576682,3094340,1041684,82173447
Layer22,1238724,8576682,3094428,1041684,82173447
Layer23,1239204,8576682,3092760,1009972,69088775
""")
