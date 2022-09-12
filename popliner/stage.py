# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

'''
This module contains the Stage class, used to describe a single pipeline stage
containing one or more operations.
'''

import sys
import logging
from itertools import repeat
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
from tqdm import tqdm
from collections import defaultdict
import numpy as np
logger = logging.getLogger("root")


class Stage:
    '''This class represents a single stage in a pipeline, consisting of a number of
    operations.'''
    max_vars_usage_cache = {}
    exchange_code_by_tile_cache = {}
    control_code_by_tile_cache = {}
    static_analysis_performed = False

    @staticmethod
    def reset_static_state():
        '''Clear all static variables.'''
        Stage.set_static_state()
        Stage.static_analysis_performed = False

    @staticmethod
    def get_static_state():
        'Returns a list containing all static variables in same order accepted by set_static_state.'
        assert Stage.static_analysis_performed
        return [Stage.step_ids,
                Stage.code_bytes_by_type,
                Stage.num_tiles,
                Stage.lowered_vars,
                Stage.equivalence_classes,
                Stage.programs_cache]

    @staticmethod
    def set_static_state(state=None):
        '''Set values for all of the static variables. Values must be provided in a list in the same
           order provided by get_static_state(). If None is provided, values are reset.'''
        [Stage.step_ids,
         Stage.code_bytes_by_type,
         Stage.num_tiles,
         Stage.lowered_vars,
         Stage.equivalence_classes,
         Stage.programs_cache] = state or [[], {}, None, {}, {}, {}]
        Stage.static_analysis_performed = True

    @staticmethod
    def perform_static_analysis(report):
        '''Populate static variables required by certain functions.'''
        assert not Stage.static_analysis_performed
        Stage.num_tiles = report.compilation.target.tilesPerIpu
        if report.compilation.target.numTiles != Stage.num_tiles:
            logger.error("It looks like your model uses replication, which is not yet supported "
                         "by PopLiner. Any split points suggested are unlikely to be valid.")
        Stage.step_ids = [step.program._id for step in  # pylint: disable=W0212
                          report.compilation.livenessProgramSteps]
        logger.info("Loading vertex code size data...")
        Stage.__cache_vertex_code_bytes(report)
        logger.info("Loading lowered variables data...")
        Stage.__cache_lowered_vars(report)
        Stage.static_analysis_performed = True

    def __init__(self, named_op=None):
        self.prog_ids = set()
        self.vertex_count_by_tile = {}
        self.first_step_index = None
        self.last_step_index = None
        self.first_step_is_in_layer = False
        self.used_vars = set()
        if named_op:
            self.add(named_op)

    @property
    def vars(self):
        '''Returns frozenset of all variable IDs used by this stage.'''
        return frozenset(self.used_vars)

    @property
    def program_ids(self):
        '''Returns frozenset of all operation IDs used by this stage.'''
        return frozenset(self.prog_ids)

    # Must be called in order when populating a stage. I.e. provide the 1st operation to be executed
    # in the 1st call, and the last operation to be executed in the last call.
    def add(self, named_operation):
        '''Add an named_operation to this pipeline stage.'''
        self.prog_ids.update(named_operation.operation.program_ids)
        if self.first_step_index is None or \
           (named_operation.is_in_layer and not self.first_step_is_in_layer):
            self.first_step_index = named_operation.operation.first_step_index
            self.first_step_is_in_layer = named_operation.is_in_layer
        self.last_step_index = named_operation.operation.last_step_index
        # Ignore variables from non-layer operations as we assume they will be distributed among
        # pipline stages when recompiled.
        if named_operation.is_in_layer:
            self.used_vars.update(named_operation.operation.used_vars)
        self.vertex_count_by_tile = {}
        return self

    def __get_vertex_count_by_tile(self):
        '''Return the vertex count for each tile.  The calculated values are
        cached so repeated calls to this function are efficient.'''
        if len(self.vertex_count_by_tile) == 0:
            for prog_id in self.program_ids:
                for vertex, counts in Stage.programs_cache[prog_id].vertex_count_by_tile.items():
                    if vertex not in self.vertex_count_by_tile:
                        self.vertex_count_by_tile.setdefault(vertex,
                                                             np.zeros(Stage.num_tiles, np.uint64))
                    self.vertex_count_by_tile[vertex] += counts[:Stage.num_tiles]
        return self.vertex_count_by_tile

    @staticmethod
    def __cache_vertex_code_bytes(report):
        '''Read and cache all vertex code bytes from the profile.'''
        assert report is not None
        # The code size of a vertex varies across tiles
        for i, tile in enumerate(tqdm(report.compilation.tiles, leave=False)):
            if i == Stage.num_tiles:  # Avoid errors when model uses replicas
                break
            for vertex in tile.memory.vertices:
                if vertex.type.name not in Stage.code_bytes_by_type:
                    Stage.code_bytes_by_type.setdefault(vertex.type.name,
                                                        np.zeros(Stage.num_tiles, np.uint64))
                Stage.code_bytes_by_type[vertex.type.name][i] = vertex.codeBytes

    def code_bytes_by_tile(self):
        '''For each tile, returns the number of code bytes of the vertex
        instances in this pipeline stage.'''
        assert Stage.static_analysis_performed
        bytes_by_tile = np.zeros(Stage.num_tiles, dtype=np.uint64)
        for (v_name, _), counts in self.__get_vertex_count_by_tile().items():
            if v_name in Stage.code_bytes_by_type:
                bytes_by_tile += \
                    Stage.code_bytes_by_type[v_name][:Stage.num_tiles] * counts.astype(bool)
        return bytes_by_tile

    def vertex_state_bytes_by_tile(self):
        '''Returns the number of vertex state bytes at each tile for this pipeline stage.'''
        bytes_by_tile = np.zeros(Stage.num_tiles, dtype=np.uint64)
        for (_, v_size), counts in self.__get_vertex_count_by_tile().items():
            bytes_by_tile += counts * v_size
        return bytes_by_tile

    def exchange_code_by_tile(self):
        '''Returns the exchange code size at each tile for this pipeline stage.'''

        prog_ids = self.program_ids
        if prog_ids not in Stage.exchange_code_by_tile_cache:
            Stage.exchange_code_by_tile_cache[prog_ids] = np.zeros(Stage.num_tiles,
                                                                   dtype=np.uint64)
            for prog_id in prog_ids:
                Stage.exchange_code_by_tile_cache[prog_ids] += \
                    Stage.programs_cache[prog_id].exchange_code_by_tile[:Stage.num_tiles]
        return Stage.exchange_code_by_tile_cache[prog_ids]

    def control_code_by_tile(self):
        '''Returns the control code size at each tile for this pipeline stage.'''

        prog_ids = self.program_ids
        if prog_ids not in Stage.control_code_by_tile_cache:
            Stage.control_code_by_tile_cache[prog_ids] = np.zeros(Stage.num_tiles,
                                                                  dtype=np.uint64)
            for prog_id in prog_ids:
                Stage.control_code_by_tile_cache[prog_ids] += \
                    Stage.programs_cache[prog_id].control_code_by_tile[:Stage.num_tiles]
        return Stage.control_code_by_tile_cache[prog_ids]

    @staticmethod
    def __cache_lowered_vars(report):
        '''Returns and caches a dictionary of brief information for all lowered vars.'''
        all_brief_vars = report.compilation.loweredVariables.allBriefVars
        if len(all_brief_vars) == 0:
            sys.exit("Profile does not contain lowered variables")
        for var in tqdm(all_brief_vars, leave=False):
            if var.id not in Stage.lowered_vars:
                Stage.lowered_vars.setdefault(
                    var.id, (var.equivalenceClass.id, np.uint64(var.bytes), var.tileId))
            if var.equivalenceClass.id not in Stage.equivalence_classes:
                eq_cls = Stage.equivalence_classes.setdefault(
                    var.equivalenceClass.id, np.zeros(len(Stage.step_ids), dtype=np.bool))
                for interval in var.equivalenceClass.steps:
                    eq_cls[interval.start:interval.end] = 1

    @staticmethod
    def tile_max_vars_usage(step_to, step_from, bytes_by_eq_class):
        '''For a particular tile returns the highest memory requirement due to
        explicit variables.'''
        assert step_to > step_from
        bytes_by_step = np.zeros(step_to - step_from, dtype=np.uint64)

        for eq_class, size in bytes_by_eq_class.items():
            bytes_by_step += Stage.equivalence_classes[eq_class][step_from:step_to] * size

        return np.max(bytes_by_step)

    def max_vars_usage(self, multiprocessor=True):
        '''For each tile returns the highest memory requirement due to explicit variables.'''
        if self.vars not in Stage.max_vars_usage_cache:
            assert Stage.static_analysis_performed

            # Aggregate variable bytes by liveness equivalence class
            bytes_by_eq_class_by_tile = [defaultdict(np.uint64) for _ in range(Stage.num_tiles)]
            for var_id in self.used_vars:
                (v_eq, v_bytes, v_tile) = Stage.lowered_vars[var_id]
                bytes_by_eq_class_by_tile[v_tile][v_eq] += v_bytes

            # Finally compute the memory requirement for every step
            with Pool(8) if multiprocessor else ThreadPool(8) as pool:  # 8 threads seems optimal
                Stage.max_vars_usage_cache[self.vars] = \
                    pool.starmap(Stage.tile_max_vars_usage, zip(repeat(self.last_step_index + 1),
                                                                repeat(self.first_step_index),
                                                                bytes_by_eq_class_by_tile))

        return Stage.max_vars_usage_cache[self.vars]
