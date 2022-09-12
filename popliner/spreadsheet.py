# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

'''
Provides spreadsheet elements such as cell formula and list of columns.
'''

from enum import Enum
import numpy as np


def operation_to_string(named_op, stage, delimiter):
    '''Returns the named_op as a string according to fields in Field.'''
    return delimiter.join([f.as_string(named_op, stage) for f in Field])


def stage_to_string(name, stage, delimiter):
    '''Returns the stage as a string according to fields in StageField.'''
    return delimiter.join([f.as_string(name, stage) for f in StageField])


class StageField(Enum):
    '''List of columns in spreadsheet in order starting from A.
    Format used: ENTRY = "<Column name>", <function to get associated value>"'''
    NAME = "Name", lambda a: a.name
    VERTEX_STATE = "Vertex state", lambda a: np.sum(a.vertex_state_bytes_by_tile())
    VERTEX_CODE = "Vertex code", lambda a: np.sum(a.code_bytes_by_tile())
    EXCHANGE_CODE = "Exchange code", lambda a: np.sum(a.exchange_code_by_tile())
    CONTROL_CODE = "Control code", lambda a: np.sum(a.control_code_by_tile())
    VARIABLES = "Variables", lambda a: np.sum(a.max_vars_usage(False))
    TOTAL = "Total (MB)", lambda a: round((np.sum(a.vertex_state_bytes_by_tile()) +
                                           np.sum(a.code_bytes_by_tile()) +
                                           np.sum(a.exchange_code_by_tile()) +
                                           np.sum(a.control_code_by_tile()) +
                                           np.sum(a.max_vars_usage())) / (1024*1024))

    def __new__(cls, *_args, **_kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, name, value_function):
        self.column_name = name
        self.value_function = value_function

    def as_string(self, name, stage):
        '''Get this field as a string.'''
        stage.name = name
        return str(self.value_function(stage))


class Field(Enum):
    '''List of columns in spreadsheet in order starting from A.
    Format used: ENTRY = "<Column name>", <function to get associated value>"'''

    LAYER = "Layer", lambda op, _: str(op.layer)
    NOTE = "Note", lambda op, _: str(op.layer_name_note)
    FULL_NAME = "Full name", lambda op, _: op.name
    VERTEX_STATE = "Vertex state", lambda _, stage: np.sum(stage.vertex_state_bytes_by_tile())
    VERTEX_CODE = "Vertex code", lambda _, stage: np.sum(stage.code_bytes_by_tile())
    EXCHANGE_CODE = "Exchange code", lambda _, stage: np.sum(stage.exchange_code_by_tile())
    CONTROL_CODE = "Control code", lambda _, stage: np.sum(stage.control_code_by_tile())
    VARIABLES = "Variables", lambda _, stage: np.sum(stage.max_vars_usage(False))

    def __new__(cls, *_args, **_kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, name, value_function):
        self.column_name = name
        self.column_letter = chr(self.value + ord('A') - 1)
        self.value_function = value_function

    def as_string(self, named_op, stage):
        '''Get this field as a string.'''
        return str(self.value_function(named_op, stage))
