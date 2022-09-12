# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

'''
This module contains OperationList, a container which stores a list of
operations.  The container also implements a function to print the
operations in CSV or TSV format.
'''

import json
import re
import logging
import pickle
from tqdm import tqdm
from pva import OperationAnalysis  # pylint: disable=no-name-in-module
from popliner.operation import Operation, NamedOperation
from popliner.stage import Stage
from popliner import spreadsheet
import popliner.parse_args

logger = logging.getLogger("root")


def log_applicable_warnings(percentage_layers_found, calls_with_multiple_input_layers_found,
                            layer_name_regex):
    '''Log any applicable warnings following analysis performed in deduce_layer_names().'''
    if percentage_layers_found < 50:
        logger.warning(
            "Only found layers for %d%% of operations. This may indicate a lack of layer names "
            "in your model or an inappropriate value for --layer-name-regex. Use the "
            "--operation-breakdown flag to see the names of operations found.",
            round(percentage_layers_found))
    else:
        logger.info("Layers found for %d%% of operations.", round(percentage_layers_found))

    if calls_with_multiple_input_layers_found:
        logger.warning(
            "Calls were found with inputs from multiple layers - which can lead to poor "
            "results. Consider recompiling your profile with the environment variable: "
            "POPART_POPLINER_OUTLINER_REGEX='%s'", layer_name_regex)


class OperationList:
    '''Container to store list of operations.'''

    @classmethod
    def from_file(cls, pickle_load_filename, args=popliner.parse_args.default_args()):
        '''Load the operation list and Stage static state from a pickle file.'''
        logger.info("Loading operations from %s...", pickle_load_filename)

        with open(pickle_load_filename, "rb") as in_file:
            [operations, stage_static_state] = pickle.load(in_file)
        Stage.set_static_state(stage_static_state)
        operations.layer_order = args.layer_order
        operations.deduce_layer_names(args.layer_name_regex, args.inputs_regex_layer_0)
        return operations

    def __init__(self, report, args=popliner.parse_args.default_args()):
        self.layer_order = args.layer_order
        Stage.reset_static_state()
        Stage.perform_static_analysis(report)

        # Map from unique ID to Operation object
        self.unique_ops = {}
        # Ordered list of NamedOperation objects
        self.named_ops = []

        logger.info("Loading operations...")

        op_analysis = OperationAnalysis(report)
        raw_ops = op_analysis.operations
        uid_to_raw_ops_index = {}
        for i in tqdm(range(len(raw_ops)), leave=False):  # pylint: disable=C0200 # See T56321
            uid = raw_ops[i].debugContext.id
            if uid not in self.unique_ops:
                self.unique_ops[uid] = Operation(raw_ops[i])
                uid_to_raw_ops_index[uid] = i
            if self.unique_ops[uid].n_progs > 0:
                name = raw_ops[i].name
                if name in ("Call", ""):  # Make easier to identify
                    op_json = json.loads(self.unique_ops[uid].debug_context_json)
                    tensor_id = op_json.get("tensorId")
                    if name == "":
                        name = "Tensor" if tensor_id else "Anonymous"
                    if op_json.get("instanceId"):
                        name += "(" + op_json.get("instanceId") + ")"
                    if tensor_id:
                        self.unique_ops[uid].is_tensor = True
                        name += "/" + tensor_id
                self.named_ops.append(NamedOperation(self, uid, name))

        logger.info("Found %d operations, %d of which have programs (%d%% unique operations).",
                    len(raw_ops), len(self.named_ops),
                    100 * sum(map(lambda op: False if op is None else
                              op.n_progs > 0, self.unique_ops.values()))/len(self.named_ops))

        self.deduce_layer_names(args.layer_name_regex, args.inputs_regex_layer_0)
        logger.info("Loading operation variables into Python objects...")

        for uid, operation in tqdm(self.unique_ops.items(), leave=False):
            # When pickling, do for all ops, not just in-layer ops. Although slower, it is necessary
            # because the user may use arguments to change the behaviour of deduce_layer_names() the
            # next time the pickle file is used - changing which operations are in-layer.
            if args.save_to_file or operation.is_ever_in_layer:
                operation.used_vars = {var._id for var in raw_ops[uid_to_raw_ops_index[uid]].vars}

        if args.save_to_file:
            with open(args.save_to_file, "wb") as outfile:
                pickle.dump([self, Stage.get_static_state()], outfile)
                logger.info("Saved operations to %s.", args.save_to_file)

    def layers(self):
        '''Return naturally-sorted list of layers for the operations.'''
        # Note: We remove '~' placeholder and move None to the start of the list
        if self.layer_order == "natural":
            return [None] + sorted({op.layer for op in self} - {None, '~'},
                                   key=lambda key: [int(value) if value.isdigit() else value
                                                    for value in re.split('([0-9]+)', key)])
        if self.layer_order == "steps":
            dup = {None, '~'}
            return [None] + [op.layer for op in self if not (op.layer in dup or dup.add(op.layer))]

        assert False, "Invalid layer_order value."

    def __len__(self):
        '''Return the number of operations in this operation list.'''
        return len(self.named_ops)

    def __getitem__(self, index):
        '''Return a single named operation by its index.'''
        return self.named_ops[index]

    def deduce_layer_names(self, layer_name_regex, inputs_regex_layer_0):
        '''Inspect the debug context layers and work out a sensible name for
        each layer which is not already named.'''
        cur_layer = None
        num_ops_processed = 0
        num_layerless_ops = 0
        calls_with_multiple_input_layers_found = False

        for index in reversed(range(len(self))):
            if self[index].operation.is_tensor:
                self[index].set_layer(None, "tensor_filter")
                continue

            num_ops_processed += 1
            layers_from_name = set(re.findall(layer_name_regex, self[index].name))
            if len(layers_from_name) == 1:
                cur_layer = self[index].set_layer(''.join(layers_from_name.pop()))
                continue

            op_json = json.loads(self[index].operation.debug_context_json)
            all_inputs = "/".join(op_json.get("inputs") or []).replace('.', '/')
            layers_from_inputs = set(re.findall(layer_name_regex, all_inputs))
            if len(layers_from_inputs) == 1:
                cur_layer = self[index].set_layer(''.join(layers_from_inputs.pop()),
                                                  "deduced_from_input_names")
            elif len(layers_from_inputs) == 0 and self[index].name.startswith("Call"):
                self[index].set_layer(cur_layer, "deduced_from_following_operation")
            elif re.search(inputs_regex_layer_0, all_inputs):
                # Use placeholder value until we can determine what the first layer is
                cur_layer = self[index].set_layer('~', "inputs_regex_layer_0")
                logger.debug("inputs_regex_layer_0 found in %s - assign layer 0", self[index].name)
            else:
                cur_layer = self[index].set_layer(None,
                                                  str(len(layers_from_inputs)) + "_input_layers")
                num_layerless_ops += 1
                if not self[index].name.startswith("Anonymous"):
                    logger.debug(
                        "No layer for op '%s' (%d programs, %d input layers)...",
                        self[index].name, self[index].operation.n_progs, len(layers_from_inputs))
                    if self[index].name.startswith("Call"):
                        calls_with_multiple_input_layers_found = True

        # Replace '~' placeholder with first layer (other than None)
        first_layer = self.layers()[1] if len(self.layers()) > 1 else None
        for index in reversed(range(len(self))):
            if self[index].layer == '~':
                self[index].set_layer(first_layer, self[index].layer_name_note)

        logger.info("Layers found: %s", str(self.layers()))
        percentage_layers_found = 100*(num_ops_processed-num_layerless_ops)/num_ops_processed
        log_applicable_warnings(percentage_layers_found, calls_with_multiple_input_layers_found,
                                layer_name_regex)

    def as_csv(self, delimiter, limit=None):
        '''Get entire operations list as a CSV or TSV string.'''
        logger.info("Generating CSV output...")
        result = ''
        for field in spreadsheet.Field:
            result += field.column_name + delimiter

        stages = {}
        ops = self.named_ops[:limit]
        previous_layer = None
        for named_op in tqdm(ops):
            if named_op.layer != previous_layer:
                previous_layer = named_op.layer
                result += "\n"
            stage = stages.get(named_op.uid) or stages.setdefault(named_op.uid, Stage(named_op))
            result += "\n" + spreadsheet.operation_to_string(named_op, stage, delimiter)
        return result

    def layers_as_csv(self, delimiter):
        '''Get entire layers list as a CSV string.'''
        logger.info("Generating CSV output...")
        csv = ''

        for field in spreadsheet.StageField:
            csv += field.column_name + delimiter
        csv += "\n"

        layers = self.layers()
        stages = {layer: Stage() for layer in layers}
        for operation in self.named_ops:
            stages[operation.layer].add(operation)

        for layer in tqdm(layers):
            csv += spreadsheet.stage_to_string(layer, stages[layer], delimiter) + '\n'
        return csv
