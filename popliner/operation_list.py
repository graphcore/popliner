# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

'''
This module contains OperationList, a container which stores a list of
operations.  The container also implements a function to print the
operations in CSV or TSV format.
'''

import json
import re
import logging
from tqdm import tqdm
from ordered_set import OrderedSet
from pva import OperationAnalysis  # pylint: disable=no-name-in-module
from popliner.operation import Operation
from popliner.stage import Stage
from popliner import spreadsheet

logger = logging.getLogger("root")


class NamedOperation:
    '''Wrapper for a unique Operation (according to uid) to add a name and type.'''
    def __init__(self, operation_list, uid, name):
        self.operation_list = operation_list
        self.uid = uid
        self.name = name
        self.layer_name = None
        self.layer_name_note = ""

    @property
    def operation(self):
        '''Returns the unwrapped (unnamed) operation for this object.'''
        assert self.uid in self.operation_list.unique_ops
        return self.operation_list.unique_ops[self.uid]


class OperationList:
    '''Container to store list of operations.'''

    def __init__(self, report):
        Stage.reset_static_state()
        Stage.perform_static_analysis(report)

        # Map from unique ID to Operation object
        self.unique_ops = {}
        # Ordered list of NamedOperation objects
        self.named_ops = []

        logger.info("Loading operations...")

        op_analysis = OperationAnalysis(report)
        raw_ops = op_analysis.operations
        for i in tqdm(range(len(raw_ops)), leave=False):  # pylint: disable=C0200 # See T56321
            uid = raw_ops[i].debugContext.id
            if uid not in self.unique_ops:
                self.unique_ops[uid] = Operation(raw_ops[i], i)
            if self.unique_ops[uid].n_progs > 0:
                self.named_ops.append(NamedOperation(self, uid, raw_ops[i].name))

        logger.info("Operations found - All: %d - With programs: %d - Also unique: %d.",
                    len(raw_ops), len(self.named_ops),
                    sum(map(lambda op: False if op is None else
                            op.n_progs > 0, self.unique_ops.values())))

        self._deduce_layer_names()

        logger.info("Layers found: %s", str(list(OrderedSet([op.layer_name for op in self]))))

        logger.info("Loading operation variables into Python objects...")

        for _, operation in tqdm(self.unique_ops.items(), leave=False):
            if operation.n_progs > 0:
                operation.fetch_vars(raw_ops[operation.raw_ops_index])

    def __len__(self):
        '''Return the number of operations in this operation list.'''
        return len(self.named_ops)

    def __getitem__(self, index):
        '''Return a single named operation by its index.'''
        return self.named_ops[index]

    def _deduce_layer_names(self):
        '''Inspect the debug context layers and work out a sensible name for
        each layer which is not already named.'''
        cur_layer = "POSTAMBLE"
        for index in reversed(range(len(self))):
            if self[index].name == "Call":
                self.unique_ops[self[index].uid].is_in_layer = True
                if cur_layer == "POSTAMBLE":
                    self.unique_ops[self[index].uid].is_in_layer = False
                    inputs = json.loads(self[index].operation.debug_context_json).get("inputs")
                    for an_input in inputs:
                        layer = re.match(r"(Layer\d+)", an_input)
                        if layer:
                            self[index].layer_name = layer.groups()[0]
                            self[index].layer_name_note = "(deduced_for_call_from_input_names)"
                            break
            else:
                for name_part in self[index].name.split('/'):
                    if name_part.startswith('block') or name_part.startswith('layer') or \
                       name_part.startswith('Layer'):
                        self[index].layer_name = name_part
                        self.unique_ops[self[index].uid].is_in_layer = True
                        break

            if self[index].layer_name is None:
                self[index].layer_name = cur_layer
                self[index].layer_name_note = "(deduced)"
            else:
                if self[index].layer_name_note == "":
                    cur_layer = self[index].layer_name

        for index in reversed(range(len(self))):
            if len(set(self[index].name.split('/')).intersection({'Embedding', 'Squad'})) > 0:
                self[index].layer_name = "POSTAMBLE"
                self[index].layer_name_note = "(keyword filter)"
                self.unique_ops[self[index].uid].is_in_layer = False

    def as_csv(self, delimiter, limit=None):
        '''Get entire operations list as a CSV or TSV string.'''
        logger.info("Generating CSV output...")
        csv = ''
        for field in spreadsheet.Field:
            csv += field.column_name + delimiter
        previous_op_name = ['']
        for index, item in enumerate(self.named_ops):
            if item.name is None or item.name == "":
                continue
            if limit is not None:
                limit -= 1
                if limit == 0:
                    break
            # Insert blank line before every new value in first column
            name_parts = item.name.split('/')
            if name_parts is None or len(name_parts) == 0:
                name_parts = ["Anonymous"]
            if name_parts[0] != previous_op_name[0]:
                csv += '\n'
                previous_op_name = name_parts

            csv += delimiter.join([f.as_string(self.named_ops, index)
                                  for f in spreadsheet.Field]) + '\n'

        return csv

    def layers_as_csv(self, delimiter):
        '''Get entire layers list as a CSV string.'''
        logger.info("Generating CSV output...")
        csv = ''

        for field in spreadsheet.LayerField:
            csv += field.column_name + delimiter
        csv += "\n"

        for layer in list(OrderedSet([op.layer_name for op in self.named_ops])):
            stage = Stage()
            for operation in self.named_ops:
                if operation.layer_name == layer:
                    stage.add(operation.operation)
            csv += delimiter.join([f.as_string(layer, stage)
                                  for f in spreadsheet.LayerField]) + '\n'

        return csv
