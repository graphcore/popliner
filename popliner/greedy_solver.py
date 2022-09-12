# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

'''
This module contains the GreedySolver class.  This implements an algorithm
which attempts to split a list of operations into stages which can fit on a
given number of IPUs.
'''

import logging
from tqdm import tqdm
import numpy as np
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


class GreedySolver:
    '''Splits an operation list to fit into multiple IPUs.
    We follow a greedy approach: we keep adding operations to an IPU
    until we reach its memory capacity. Then continue with the next IPU, etc.
    To achieve a balanced distribution, we first aim for X% of IPU capacity,
    and if the model doesn't fit, we increase that memory proportion.'''

    def __init__(self, named_operations, layer_operations_only=False):
        if layer_operations_only:
            self.named_operations = [op for op in named_operations if op.is_in_layer]
        else:
            self.named_operations = named_operations
        self.splits = []
        self.layers = named_operations.layers()

    def __get_block_ends(self, op_from):
        '''Returns a generator to iterate over the last operation of each block,
        starting at op_from.'''
        if len(self.named_operations) == 0:
            return
        prev_layer = self.named_operations[op_from].layer
        for i in range(op_from + 1, len(self.named_operations)):
            layer = self.named_operations[i].layer
            if layer is not None and layer != prev_layer:
                prev_layer = layer
                yield i
        yield len(self.named_operations)

    def get_memory_for_layers(self, layers=None):
        '''Returns the memory required by a subset of operations.
        Note that it's usually less than the addition of each individual
        operation due to code reuse. If layers is None, all layers are used.'''
        stage = Stage()
        for operation in self.named_operations:
            if operation.layer in (layers or self.layers):
                stage.add(operation)

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
            # Use for testing
            "stage": stage,
        }

    def __calculate_stage(self, layer_from, mem_per_tile, min_layers_per_stage):
        '''Returns how many operations fit in memory starting at layer_from.'''
        prev_layer_to = layer_from
        prev_mem = None
        # Allow less than min_layers_per_stage for final stage:
        for layer_to in range(layer_from + min_layers_per_stage - 1, len(self.layers)) or \
                range(layer_from, len(self.layers)):
            layers = self.layers[layer_from:layer_to + 1]
            mem = self.get_memory_for_layers(layers)
            max_tile_mem = np.max(mem["total_mem"])
            msg = f"{layers} Peak: {round(max_tile_mem/1024)} KB "
            if max_tile_mem > mem_per_tile:
                logger.debug("%s- OVERFLOW", '\u0336'.join(msg))  # Strikethrough
                break
            logger.debug("%s", msg)
            prev_layer_to = layer_to
            prev_mem = mem
        return {
            "layer_to": prev_layer_to,
            "mem": prev_mem,
        }

    def solve(self, num_ipus, mem_per_tile):
        '''Splits operations to fit into multiple IPU. Returns True if successfully split,
        False otherwise.'''
        self.splits.clear()
        if len(self.named_operations) == 0:
            return self.splits

        # Try to distribute memory as equally as possible among IPUs
        for min_layers_per_stage in [3, 2, 1]:
            for proportion in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                mem = mem_per_tile * proportion
                logger.info("Targeting tile capacity: %d KB (%d%%) - minimum layers: %d...",
                            mem/1024, 100*proportion, min_layers_per_stage)
                self.splits.clear()
                layer_from = 0
                for _ in tqdm(range(num_ipus), desc="Calculating stages", leave=False):
                    split = self.__calculate_stage(layer_from, mem, min_layers_per_stage)
                    if split["mem"] is None:
                        break

                    split["layer_from"] = layer_from
                    layer_from = split["layer_to"] + 1
                    self.splits.append(split)
                    if layer_from >= len(self.layers):
                        # This memory proportion worked.
                        return proportion

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
                    "max_tile_mem": int(np.max(split["mem"]["total_mem"])),
                    "variables": int(np.sum(split["mem"]["variables"])),
                    "vertex_code": int(np.sum(split["mem"]["vertex_code"])),
                    "vertex_state": int(np.sum(split["mem"]["vertex_state"])),
                    "exchange_code": int(np.sum(split["mem"]["exchange_code"])),
                    "control_code": int(np.sum(split["mem"]["control_code"]))}
            })
        return splits

    def calculate_memory_affinity(self):
        '''Returns the size of the variables shared between each two layers.'''
        assert Stage.static_analysis_performed
        vars_by_layer = []
        op_from = 0
        for op_to in self.__get_block_ends(op_from):
            logger.info("Operations: [%d, %d)", op_from, op_to)
            stage = Stage()
            for index in range(op_from, op_to):
                stage.add(self.named_operations[index])
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
                    total_bytes += Stage.lowered_vars[var_id].bytes
                print(total_bytes, "\t", end='')
            print("")

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
                stage.add(self.named_operations[index])
            steps_by_layer.append((
                self.named_operations[op_from].operation.first_step_index,
                # not exactly correct but we don't have more programs
                self.named_operations[op_to - 1].operation.last_step_index \
                if op_to == len(self.named_operations) \
                else self.named_operations[op_to].operation.first_step_index))
            logger.info("Steps: [%d, %d)", steps_by_layer[-1][0], steps_by_layer[-1][1])
            vars_by_layer.append(stage.used_vars)
            op_from = op_to

        for i, layer in enumerate(vars_by_layer):
            print("Layer: ", i, "\t", end='')
            for j, _ in enumerate(vars_by_layer):
                total_bytes = 0
                # For those variables used in both layers
                for var_id in layer.intersection(vars_by_layer[j]):
                    var = Stage.lowered_vars[var_id]
                    # If var is written in i and read in j
                    if _is_bridge(var.equivalenceClass, steps_by_layer[i], steps_by_layer[j]):
                        total_bytes += var.bytes
                print(total_bytes, "\t", end='')
            print("")
