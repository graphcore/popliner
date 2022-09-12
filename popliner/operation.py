# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from collections import namedtuple
import pva
import numpy as np
from popliner.stage import Stage

'''
This module contains the Operation class, which represents a unique operation
according to debug context ID.
'''

Program = namedtuple('Program',
                     ['control_code_by_tile', 'exchange_code_by_tile', 'vertex_count_by_tile'])


def _get_program(raw):
    assert Stage.static_analysis_performed

    # As the control code of a program includes its children's,
    # we only consider childless programs to avoid counting
    # control code twice. This means we ignore the control code
    # of programs like Sequence and Repeat but it is negligible.
    control_code_by_tile = np.array(raw.controlCodeByTile, dtype=np.uint64) \
        if (len(raw.children) == 0) else np.zeros(Stage.num_tiles, dtype=np.uint64)

    # pylint: disable=E1101
    exchange_code_by_tile = np.array(raw.codeBytesByTile, dtype=np.uint64) if raw.type == \
        pva.Program.Type.DoExchange else np.zeros(Stage.num_tiles, dtype=np.uint64)

    vertex_count_by_tile = {}
    if raw.type == pva.Program.Type.OnTileExecute:
        for vertex in raw.computeset.vertices:
            vertex_count_by_tile[(vertex.type.name, np.uint64(vertex.type.size))] = \
                np.array(vertex.countByTile, dtype=np.uint64)

    return Program(control_code_by_tile, exchange_code_by_tile, vertex_count_by_tile)


class Operation():
    '''This class represents an operation -- a collection of poplar programs
    grouped based on their debug context.'''

    def __init__(self, raw_op, used_vars=None):
        self.is_ever_in_layer = False
        self.is_tensor = False
        progs = raw_op.programs()
        self.n_progs = len(progs)
        if self.n_progs == 0:
            return
        self.program_ids = set()
        for prog in progs:
            self.program_ids.add(prog._id)
            if prog._id not in Stage.programs_cache:
                Stage.programs_cache[prog._id] = _get_program(prog)

        self.first_step_index = raw_op.firstStepIndex
        self.last_step_index = raw_op.lastStepIndex
        self.debug_context_json = raw_op.debugContext.json
        self.used_vars = set()
        if used_vars:
            self.used_vars.update(used_vars)


class NamedOperation():
    '''Wrapper for a unique Operation (according to uid) to add a name and layer.'''
    def __init__(self, operation_list, uid, name):
        self.unique_ops = operation_list.unique_ops
        self.uid = uid
        self.name = name
        self._layer = None
        self.layer_name_note = ""

    @property
    def operation(self):
        '''Returns the unwrapped (unnamed) operation for this object.'''
        return self.unique_ops[self.uid]

    def set_layer(self, layer, note=""):
        '''Set layer of this operation with explanation note. Returns the layer set.'''
        if layer is not None:
            self.operation.is_ever_in_layer = True
        self._layer = layer
        self.layer_name_note = note
        return self._layer

    @property
    def layer(self):
        '''Returns the layer of this operation.'''
        return self._layer

    @property
    def is_in_layer(self):
        '''Returns true if a layer has been set for this operation. Otherwise, false.'''
        return self.layer is not None
