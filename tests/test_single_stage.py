# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from popliner.operation_list import OperationList
from popliner.greedy_solver import GreedySolver
from popliner.stage import Stage
import pva  # pylint: disable=no-name-in-module
import pytest
import logging
import argparse
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def check_expectation(expectations, key, value):
    if key in expectations.keys():
        assert expectations[key] >= value


# Gather all unique compute sets and exchanges.
# We could do the same with the rest of Program types but these two are the
# most important for memory analysis
class ProgCounterVisitor(pva.ProgramVisitor):

    def __init__(self):
        super().__init__()
        # Set without duplicates
        self.onTileExecutes = set()
        self.doExchanges = set()

    def visitOnTileExecute(self, onTileExecute):
        self.onTileExecutes.add(onTileExecute)

    def visitDoExchange(self, doExchange):
        self.doExchanges.add(doExchange)


def print_diff(name, libpva, popliner):
    if libpva:
        diff = round((popliner - libpva) / libpva * 100, 2)
    else:
        diff = 100
    LOGGER.info(name + " diff: " + str(diff) + "% (" + str(libpva) + " (libpva) vs " +
                str(popliner) + " (popliner))")
    return diff


# This function detects programs that Popliner ignores.
# The main reason for ignoring a program is not belonging to an
# operation/layer (according to their Debug Info)
def check_programs(report, popliner, expectations):
    # Count all programs via libpva
    libpva_visitor = ProgCounterVisitor()
    for prog in report.compilation.programs:
        prog.accept(libpva_visitor)
    assert len(libpva_visitor.onTileExecutes) > 0
    assert len(libpva_visitor.doExchanges) > 0

    # Count the programs Popliner takes into account
    popliner_visitor = ProgCounterVisitor()
    for prog in popliner["stage"].programs:
        prog.raw_program.accept(popliner_visitor)

    # libpva should return all programs
    assert len(libpva_visitor.onTileExecutes) >= len(popliner_visitor.onTileExecutes)
    assert len(libpva_visitor.doExchanges) >= len(popliner_visitor.doExchanges)

    # Compare onTileExecute
    num_on_tile_execute_diff = print_diff("onTileExecutes", len(libpva_visitor.onTileExecutes),
                                          len(popliner_visitor.onTileExecutes))
    if num_on_tile_execute_diff and LOGGER.isEnabledFor(logging.DEBUG):
        missing = libpva_visitor.onTileExecutes - popliner_visitor.onTileExecutes
        LOGGER.debug("missing onTileExecutes (showing 10 out of {}):".format(len(missing)))
        for i, prog in enumerate(missing):
            if i >= 10:
                break
            LOGGER.debug("  " + prog.name)

    # Compare doExchange
    num_do_exchange_diff = print_diff("doExchange", len(libpva_visitor.doExchanges),
                                      len(popliner_visitor.doExchanges))
    if num_do_exchange_diff and LOGGER.isEnabledFor(logging.DEBUG):
        missing = libpva_visitor.doExchanges - popliner_visitor.doExchanges
        LOGGER.debug("missing doExchanges (showing 10 out of {}):".format(len(missing)))
        for i, prog in enumerate(missing):
            if i >= 10:
                break
            LOGGER.debug("  " + prog.name)

    check_expectation(expectations, "num_on_tile_execute_diff", abs(num_on_tile_execute_diff))
    check_expectation(expectations, "num_do_exchange_diff", abs(num_do_exchange_diff))


# This function checks the size of code memory extracted by Popliner
def check_code_memory(report, popliner, expectations):

    def check_overestimation(category, libpva, tile):
        # Overestimations should never happen
        assert libpva >= popliner[category][tile], f"Overestimation of {category} in tile {tile}"

    vertexCode = 0
    exchangeCode = 0
    controlCode = 0
    vertexState = 0

    for i, tile in enumerate(report.compilation.tiles):
        cat = tile.memory.category
        vertexCode += cat.vertexCode.total
        check_overestimation("vertex_code", cat.vertexCode.total, i)
        exchangeCode += cat.internalExchangeCode.total
        check_overestimation("exchange_code", cat.internalExchangeCode.total, i)
        controlCode += cat.controlCode.total
        check_overestimation("control_code", cat.controlCode.total, i)
        vertexStateAcc = cat.vertexInstanceState.total
        vertexStateAcc += cat.vectorListDescriptor.total
        vertexState += vertexStateAcc
        check_overestimation("vertex_state", vertexStateAcc, i)

    vertex_code_diff = print_diff("vertex code", vertexCode, sum(popliner["vertex_code"]))
    vertex_state_diff = print_diff("vertex state", vertexState, sum(popliner["vertex_state"]))
    control_code_diff = print_diff("control code", controlCode, sum(popliner["control_code"]))
    # Note libpva exchange code aggregation differs from the addition of each exchange code
    # size possibly due to code sharing among exchanges
    exchange_code_diff = print_diff("exchange code", exchangeCode, sum(popliner["exchange_code"]))

    check_expectation(expectations, "vertex_code_diff", abs(vertex_code_diff))
    check_expectation(expectations, "vertex_state_diff", abs(vertex_state_diff))
    check_expectation(expectations, "control_code_diff", abs(control_code_diff))
    check_expectation(expectations, "exchange_code_diff", abs(exchange_code_diff))


def calculate_max_memory(report, step_from, step_to, tile):
    max = 0
    max_step = 0
    max_vars = 0
    for i, step in enumerate(report.compilation.livenessProgramSteps):
        if i < step_from:
            continue
        elif i >= step_to:
            break
        mem = step.notAlwaysLiveMemoryForTile(tile).bytes
        if max < mem:
            max = mem
            max_step = i
            max_vars = step.notAlwaysLiveMemoryForTile(tile).variables
    return max, max_step, max_vars


def get_consumers(report, var_id):
    progs = []
    for prog in report.compilation.programs:
        for var in prog.vars:
            if var._id == var_id:
                progs.append(prog)
    return progs


def is_live(eq_class, step):
    return Stage.equivalence_classes[eq_class][step]


# This function identifies the memory bottleneck (tile and step)
# and compares libpva and popliner at that point
def check_memory_bottleneck(report, popliner, expectations):
    # Let's find the tile with highest memory requirement according to libpva.
    # That is the memory bottleneck and what determines if a stage fits in an IPU.
    # But let's ignore the steps that Popliner does not take into account
    # (probably those operations outside any layer).
    # For that we have to recalculate max liveness for tile

    step_from = popliner["stage"].first_step_index
    step_to = popliner["stage"].last_step_index + 1
    LOGGER.info("Popliner steps from {} to {}".format(step_from, step_to))
    total_num_steps = len(report.compilation.livenessProgramSteps)
    step_diff = print_diff("Steps", total_num_steps, step_to - step_from)
    check_expectation(expectations, "steps_diff", abs(step_diff))

    num_tiles = report.compilation.target.numTiles
    tiles = range(0, num_tiles)
    # Calculating all tiles takes some time so focus on the bottleneck if we already know it
    if "worst_tile" in expectations.keys():
        tile = expectations["worst_tile"]
        tiles = range(tile, tile + 1)

    max = 0
    max_tile = 0
    max_step = 0
    max_vars = 0
    # TODO: parallelise
    pbar = tqdm(tiles)
    for tile in pbar:
        pbar.set_description("Analysing bottleneck tile")
        local_max, local_max_step, local_max_vars = \
            calculate_max_memory(report, step_from, step_to, tile)
        local_max += report.compilation.tiles[tile].memory.alwaysLiveBytes
        if max < local_max:
            max = local_max
            max_tile = tile
            max_step = local_max_step
            max_vars = local_max_vars

    LOGGER.info("Memory bottleneck in tile {} and step {}".format(max_tile, max_step))
    mem_diff = print_diff("Memory at bottleneck tile", max, popliner["total_mem"][max_tile])
    assert max >= popliner["total_mem"][max_tile], "Memory overestimation"
    check_expectation(expectations, "worst_tile_step_mem_diff", abs(mem_diff))

    if LOGGER.isEnabledFor(logging.DEBUG):
        # Compare code sizes between popliner and libpva
        LOGGER.debug("Popliner vertex code: {}B".format(popliner["vertex_code"][max_tile]))
        LOGGER.debug("Popliner vertex state: {}B".format(popliner["vertex_state"][max_tile]))
        LOGGER.debug("Popliner exchange code: {}B".format(popliner["exchange_code"][max_tile]))
        LOGGER.debug("Popliner control code: {}B".format(popliner["control_code"][max_tile]))

        key = "alwaysLiveVariablesForTile"
        if hasattr(report.compilation, key) and callable(getattr(report.compilation, key)):
            always_live_vars_dict = {}
            for var in report.compilation.alwaysLiveVariablesForTile(max_tile):
                always_live_vars_dict[var.name] = var
            code_names = ["vertexCode", "vertexInstanceState", "internalExchangeCode",
                          "controlCode", "vertexFieldData", "hostExchangeCode"]
            for name in code_names:
                assert name in always_live_vars_dict
                LOGGER.debug("libpva {}: {}B".format(name, always_live_vars_dict[name].size))
        else:
            LOGGER.warning("Skipping always-live variables due to old Poplar SDK.".format(key))

        # We cannot correlate libpva and popliner vars using their ids because libpva vars are
        # extracted from the liveness analysis and it groups variables by unlowered id while
        # popliner uses lowered variables.
        # Moreover, some libpva unlowered ids are artificial if they belong to a group of
        # "low-level" variables like Poplar or Poplibs variables. This means we cannot extract
        # lowered variables from every unlowered id generated during liveness analysis.
        # We will do a best-effort approach: among the variables that have a valid unlowered var id,
        # filter those that are present in libpva and popliner. For the rest, print a list and
        # let the user interpret the result.

        # Collect "unlowered" variables from libpva not-always and always live analysis
        libpva_vars = {}
        var_mem_acc = 0
        for var in max_vars:
            libpva_vars[var] = var
            var_mem_acc += var.size
        key = "alwaysLiveVariablesForTile"
        if hasattr(report.compilation, key) and callable(getattr(report.compilation, key)):
            for var in report.compilation.alwaysLiveVariablesForTile(max_tile):
                var_mem_acc += var.size
                # Code variables already listed above
                if var.name not in code_names:
                    libpva_vars[var] = var
            assert var_mem_acc == max

        # Now collect from popliner the used variables at the bottleneck tile that are live
        # in the bottleneck step
        popliner_vars = []
        size_common = 0
        # As popliner only keeps partial information of the used variables we will need to
        # complement it with information from the full lowered variable list
        lowered_vars = report.compilation.loweredVariables.forTile(max_tile)
        lowered_vars = {var._id: var for var in lowered_vars}
        for var_id in popliner["stage"].used_vars:
            (v_eq, v_bytes, v_tile) = Stage.lowered_vars[var_id]
            if v_tile == max_tile and is_live(v_eq, max_step):
                # Let's focus on the disparities so filter out the variables present in libpva
                # and popliner
                if lowered_vars[var_id].hasUnloweredVar:
                    unlowered_var = lowered_vars[var_id].unloweredVar
                    if unlowered_var in libpva_vars:
                        assert lowered_vars[var_id].bytes == libpva_vars[unlowered_var].size
                        size_common += lowered_vars[var_id].bytes
                        del libpva_vars[unlowered_var]
                        continue
                popliner_vars.append([var_id, v_bytes])
        LOGGER.debug("Some variables present in both libpva and popliner are not shown ({}B)"
                     .format(size_common))

        num_vars_to_list = 20
        LOGGER.debug("{} largest variables from Popliner".format(num_vars_to_list))
        # Finally sort by size and name
        popliner_vars.sort(key=lambda x: (x[1], lowered_vars[x[0]].name), reverse=True)
        for var in popliner_vars[:num_vars_to_list]:
            LOGGER.debug("  {}: {}B".format(lowered_vars[var[0]].name, var[1]))
            if lowered_vars[var[0]].hasUnloweredVar:
                lowered_vars[var[0]].unloweredVar

        LOGGER.debug("{} largest variables from libpva".format(num_vars_to_list))
        libpva_vars = list(libpva_vars.values())
        libpva_vars.sort(key=lambda x: (x.size, x.name), reverse=True)
        for var in libpva_vars[:num_vars_to_list]:
            LOGGER.debug("  {}: {}B".format(var.name, var.size))


# In this test we ask the solver to create one single stage with all the operations of the model.
# In practice we would allow the solver to split the model into multiple stages
# but here we want to check that the memory analysis done by the solver is similar to the actual
# memory used by the model and, in particular, that the solver is not missing any operation or
# variable.
def check_single_stage(report, expectations, layer_operations_only=False):
    operations = OperationList(report)
    solver = GreedySolver(report, operations, layer_operations_only)

    # Get one single stage rather than splitting the model into multiple
    total = solver.get_single_stage_mem_for_inference()

    # The model must be single IPU
    assert len(total["total_mem"]) == 1472

    check_programs(report, total, expectations)

    check_code_memory(report, total, expectations)

    check_memory_bottleneck(report, total, expectations)


def test_single_stage_popart_bert_base():
    # diffs are represented in percentage
    expectations = {
        "num_on_tile_execute_diff": 78.57,
        "num_do_exchange_diff": 86.79,
        "vertex_code_diff": 50.48,
        "vertex_state_diff": 64.34,
        "control_code_diff": 62.21,
        "exchange_code_diff": 82.88,
        "steps_diff": 63.57,
        "worst_tile": 632,
        # This is the most important metric, the memory
        # deviation at the bottleneck (tile and step)
        "worst_tile_step_mem_diff": 16.59,
    }
    # If non-layer operations are also considered
    expectations_including_non_layer_operations = {
        "num_on_tile_execute_diff": 0.55,
        "num_do_exchange_diff": 0.0,
        "vertex_code_diff": 0.8,
        "vertex_state_diff": 4.72,
        "exchange_code_diff": 6.46,
        "control_code_diff": 5.16,
        "steps_diff": 1.0,
        "worst_tile": 632,
        "worst_tile_step_mem_diff": 10.76,
    }
    report = pva.openReport('tests/reports/popart_bert_train_base/profile.pop',
                            'tests/reports/popart_bert_train_base/debug.cbor')
    layer_operations_only = True
    check_single_stage(report, expectations, layer_operations_only)


@pytest.mark.skip(reason="Tensorflow not ready")
def test_single_stage_tf_bert_tiny():
    expectations = {}
    report = pva.openReport('tests/reports/tf_bert_train_tiny/profile.pop',
                            'tests/reports/tf_bert_train_tiny/debug.cbor')
    check_single_stage(report, expectations)


@pytest.mark.skip(reason="Tensorflow not ready")
def test_single_stage_tf_bert_base():
    expectations = {}
    report = pva.openReport('tests/reports/tf_bert_train_base/profile.pop',
                            'tests/reports/tf_bert_train_base/debug.cbor')
    check_single_stage(report, expectations)


def main():
    parser = argparse.ArgumentParser(
              description='A tool to analyse the memory accuracy of Popliner')
    parser.add_argument('-p', '--profile', type=str, required=True,
                        help='The path to profile.pop')
    parser.add_argument('-d', '--debug', type=str, required=True,
                        help='The path to debug.cbot')
    parser.add_argument('--layer-operations-only', type=bool, default=False,
                        help='The path to debug.cbot')
    args = parser.parse_args()

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s\t%(message)s')
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG)

    expectations = {}
    report = pva.openReport(args.profile, args.debug)
    check_single_stage(report, expectations, args.layer_operations_only)


if __name__ == '__main__':
    main()
