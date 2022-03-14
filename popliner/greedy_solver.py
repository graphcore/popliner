# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

'''
This module contains the GreedySolver class.  This implements an algorithm
which attempts to split a list of operations into stages which can fit on a
given number of IPUs.
'''

import logging
import json
from tqdm import tqdm
import numpy as np
from ordered_set import OrderedSet
from popliner.stage import Stage

logger = logging.getLogger("root")


def _is_bridge(eq_class, steps1, steps2):
    '''Returns true if the equivalence class goes live during steps1 and dies
    during steps2.'''
    for interval in eq_class.steps:
        if interval.start >= steps1[0] and interval.start <= steps1[1] and \
           interval.end >= steps2[0] and interval.end < steps2[1]:
            return True
    return False


class GreedySolver:  # pylint: disable=too-few-public-methods
    '''Splits an operation list to fit into multiple IPUs.
    We follow a greedy approach: we keep adding operations to an IPU
    until we reach its memory capacity. Then continue with the next IPU, etc.
    To achieve a balanced distribution, we first aim for X% of IPU capacity,
    and if the model doesn't fit, we increase that memory proportion.'''

    def __init__(self, report, operations):
        self.report = report
        self.operations = operations
        self.splits = []
        self.layers = list(OrderedSet([op.layer_name for op in self.operations]))

    def __get_block_ends(self, op_from):
        '''Returns a generator to iterate over the last operation of each block,
        starting at op_from.'''
        if len(self.operations) == 0:
            return
        prev_name = self.operations[op_from].layer_name
        for i in range(op_from + 1, len(self.operations)):
            name = self.operations[i].layer_name
            if name is not None and name != prev_name:
                prev_name = name
                yield i
        yield len(self.operations)

    def __get_stage_mem_for_inference(self, layer_from, layer_to):
        '''Returns the memory required by a subset of operations.
        Note that it's usually less than the addition of each individual
        operation due to code reuse.'''
        layers = self.layers[layer_from:layer_to + 1]
        logger.debug("Calculating stage %d-%d - layers: %s", layer_from, layer_to, str(layers))
        stage = Stage()
        for operation in self.operations:
            if operation.layer_name in layers:
                stage.add(operation.operation)

        exchange_code = stage.exchange_code_by_tile()
        vertex_state = stage.vertex_state_bytes_by_tile()
        vertex_code = stage.code_bytes_by_tile()
        control_code = stage.control_code_by_tile()
        variables = stage.max_vars_usage()
        zipped = zip(variables, vertex_code, vertex_state, exchange_code, control_code)
        total_mem = [sum(item) for item in zipped]
        return {
            "total_mem": total_mem,
            "variables": variables,
            "vertex_code": vertex_code,
            "vertex_state": vertex_state,
            "exchange_code": exchange_code,
            "control_code": control_code,
        }

    def __calculate_stage(self, layer_from, mem_per_tile, min_layers_per_stage):
        '''Returns how many operations fit in memory starting at layer_from.'''
        prev_layer_to = layer_from
        prev_mem = None
        for layer_to in range(layer_from + min_layers_per_stage - 1, len(self.layers)):
            mem = self.__get_stage_mem_for_inference(layer_from, layer_to)
            if max(mem["total_mem"]) > mem_per_tile:
                break
            prev_layer_to = layer_to
            prev_mem = mem
        return {
            "layer_to": prev_layer_to,
            "mem": prev_mem,
        }

    def get_single_stage_mem_for_inference(self):
        '''Returns a memory breakdown by category of a single stage containing all operations.'''
        return self.__get_stage_mem_for_inference(0, len(self.layers) - 1)

    def solve(self, num_ipus, mem_per_tile):
        '''Splits operations to fit into multiple IPU. Returns True if successfully split,
        False otherwise.'''
        self.splits.clear()
        if len(self.operations) == 0:
            return self.splits

        # Try to distribute memory as equally as possible among IPUs
        for min_layers_per_stage in [3, 2, 1]:
            for proportion in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                logger.info("Targeting memory proportion: %f, minimum layers per stage: %d...",
                            proportion, min_layers_per_stage)
                self.splits.clear()
                layer_from = 0
                mem = mem_per_tile * proportion
                for _ in tqdm(range(num_ipus), desc="Calculating stages", leave=False):
                    split = self.__calculate_stage(layer_from, mem, min_layers_per_stage)
                    if split["mem"] is None:
                        break

                    split["layer_from"] = layer_from
                    layer_from = split["layer_to"] + 1
                    self.splits.append(split)
                    if layer_from >= len(self.layers):
                        # This memory proportion worked.
                        return True

        # Even filling every IPU to 100% capacity, we couldn't fit the
        # operations on this many IPUs - try increasing num_ipus.
        return False

    def get_splits_totals(self):
        '''Returns split information with memory values summed for all tiles.'''
        splits = []
        for split in self.splits:
            splits.append({
                "layer_from": self.layers[split["layer_from"]],
                # Re-calculate op_to to ignore all out-of-layer operations
                "layer_to": self.layers[split["layer_to"]],
                "mem": {
                    "total_mem": int(np.sum(split["mem"]["total_mem"])),
                    "variables": int(np.sum(split["mem"]["variables"])),
                    "vertex_code": int(np.sum(split["mem"]["vertex_code"])),
                    "vertex_state": int(np.sum(split["mem"]["vertex_state"])),
                    "exchange_code": int(np.sum(split["mem"]["exchange_code"])),
                    "control_code": int(np.sum(split["mem"]["control_code"]))}
            })
        return splits

    def get_splits_as_json(self):
        '''Returns a JSON dump of the split information, with memory values summed for all tiles.'''
        return json.dumps(self.get_splits_totals(), indent=4)

    def calculate_memory_affinity(self):
        '''Returns the size of the variables shared between each two layers.'''
        assert Stage.static_analysis_performed
        lowered_vars = Stage.lowered_vars
        vars_by_layer = []
        op_from = 0
        for op_to in self.__get_block_ends(op_from):
            logger.info("Operations: [%d, %d)", op_from, op_to)
            stage = Stage()
            for index in range(op_from, op_to):
                stage.add(self.operations[index])
            op_from = op_to
            # Aggregate variables in all tiles
            all_vars = set()
            for tile_vars in stage.used_vars:
                all_vars.update(tile_vars)
            vars_by_layer.append(all_vars)

        for i, layer in enumerate(vars_by_layer):
            logger.info(len(layer))
            print("Layer: ", i, "\t", end='')
            for j, _ in enumerate(vars_by_layer):
                total_bytes = 0
                for var_id in layer.intersection(vars_by_layer[j]):
                    total_bytes += lowered_vars[var_id].bytes
                print(total_bytes, "\t", end='')
            print("")

    def _compilation_steps(self, prog1, prog2):
        '''Returns the step indices corresponding to those programs.'''
        found = 0
        for i, step in enumerate(self.report.compilation.livenessProgramSteps):
            if step.program == prog1:
                step1 = i
                found = 1
            if step.program == prog2:
                step2 = i
                assert found == 1
                found = 2
                break
        assert found == 2
        return (step1, step2)

    def calculate_interlayer_exchange(self):
        '''For each pair of layers, calculate the size of variables that are created in one layer
           (variable goes live) and consumed by the other layer (variable dies).'''
        assert Stage.static_analysis_performed
        vars_by_layer = []
        steps_by_layer = []
        op_from = 0
        for op_to in self.__get_block_ends(op_from):
            logger.info("Operations: [%d, %d)", op_from, op_to)
            stage = Stage()
            for index in range(op_from, op_to):
                stage.add(self.operations[index])
            self.operations[op_from].get_programs()
            first_prog = self.operations[op_from].first_prog
            if op_to == len(self.operations):
                self.operations[op_to - 1].get_programs()
                # Not exactly correct but we don't have more programs
                next_first_prog = self.operations[op_to - 1].last_prog
            else:
                self.operations[op_to].get_programs()
                next_first_prog = self.operations[op_to].first_prog
            first_step, next_first_step = self._compilation_steps(first_prog, next_first_prog)
            op_from = op_to
            logger.info("Steps: [%d, %d)", first_step, next_first_step)
            steps_by_layer.append((first_step, next_first_step))
            # Aggregate variables in all tiles
            all_vars = set()
            for tile_vars in stage.used_vars:
                all_vars.update(tile_vars)
            vars_by_layer.append(all_vars)

        lowered_vars = Stage.lowered_vars
        for i, layer in enumerate(vars_by_layer):
            print("Layer: ", i, "\t", end='')
            for j, _ in enumerate(vars_by_layer):
                total_bytes = 0
                # For those variables used in both layers
                for var_id in layer.intersection(vars_by_layer[j]):
                    var = lowered_vars[var_id]
                    # If var is written in i and read in j
                    if _is_bridge(var.equivalenceClass, steps_by_layer[i], steps_by_layer[j]):
                        total_bytes += var.bytes
                print(total_bytes, "\t", end='')
            print("")
