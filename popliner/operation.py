# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pva
import numpy as np
from popliner.stage import Stage

'''
This module contains the Operation class, which represents a unique operation
according to debug context ID.
'''


class Program():  # pylint: disable=R0903
    '''Class to hold information about a single Poplar program.'''

    def __init__(self, raw):
        assert Stage.static_analysis_performed

        # As the control code of a program includes its children's,
        # we only consider childless programs to avoid counting
        # control code twice. This means we ignore the control code
        # of programs like Sequence and Repeat but it is negligible.
        self.control_code_by_tile = np.array(raw.controlCodeByTile, dtype=np.uint64) \
            if (len(raw.children) == 0) else np.zeros(Stage.num_tiles, dtype=np.uint64)

        # pylint: disable=E1101
        self.exchange_code_by_tile = np.array(raw.codeBytesByTile, dtype=np.uint64) if raw.type == \
            pva.Program.Type.DoExchange else np.zeros(Stage.num_tiles, dtype=np.uint64)

        self.vertex_count_by_tile = {}
        if raw.type == pva.Program.Type.OnTileExecute:
            for vertex in raw.computeset.vertices:
                self.vertex_count_by_tile[(vertex.type.name, np.uint64(vertex.type.size))] = \
                    np.array(vertex.countByTile, dtype=np.uint64)


class Operation():
    '''This class represents an operation -- a collection of poplar programs
    grouped based on their debug context.'''
    programs = {}

    def __init__(self, raw_op, raw_ops_index):
        self.raw_ops_index = raw_ops_index
        progs = raw_op.programs()
        self.n_progs = len(progs)
        if self.n_progs == 0:
            return
        self.programs = set()
        self.program_ids = set()
        for prog in progs:
            cached_prog = Operation.programs.setdefault(prog._id, Program(prog))
            self.programs.add(cached_prog)
            self.program_ids.add(prog._id)

        self.is_in_layer = False
        self.first_step_index = raw_op.firstStepIndex
        self.last_step_index = raw_op.lastStepIndex
        self.debug_context_json = raw_op.debugContext.json
        self.stage = None
        self.used_vars = None

    def fetch_vars(self, raw_op):
        '''Fetch used variables from raw op. Must be called before self.used_vars can be used.'''
        # Ignore variables from non-layer operations as we assume they will be distributed among
        # pipline stages when recompiled.
        if self.is_in_layer:
            self.used_vars = set(var._id for var in raw_op.vars)  # pylint: disable=W0212
        else:
            self.used_vars = set()

    def __stage(self):
        '''If this operation is already a member of a pipeline stage, return
        that stage.  Otherwise, create a new pipeline stage, add this
        operation to it, and return the new stage.'''
        if self.stage is None:
            self.stage = Stage().add(self)
        return self.stage

    def vertex_code_bytes(self):
        '''Returns the number of code bytes of the vertex instances.'''
        return np.sum(self.__stage().code_bytes_by_tile())

    def vertex_state_bytes(self):
        '''Returns the number of vertex state bytes for this operation.'''
        return np.sum(self.__stage().vertex_state_bytes_by_tile())

    def exchange_code(self):
        '''Returns the exchange code size for this operation.'''
        return np.sum(self.__stage().exchange_code_by_tile())

    def variable_bytes(self):
        '''Returns both activations and weights.'''
        return np.sum(self.__stage().max_vars_usage())

    def control_code_bytes(self):
        '''Return the number of control code bytes for this operation.'''
        return np.sum(self.__stage().control_code_by_tile())
