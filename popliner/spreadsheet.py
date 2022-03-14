# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

'''
Provides spreadsheet elements such as cell formula and list of columns.
'''

from enum import Enum
import numpy as np


class LayerField(Enum):
    '''List of columns in spreadsheet in order starting from A.
    Format used: ENTRY = "<Column name>", <function to get associated value>"'''
    LAYER = "Layer", lambda a: a.name
    VERTEX_STATE = "Vertex state", lambda a: np.sum(a.vertex_state_bytes_by_tile())
    VERTEX_CODE = "Vertex code", lambda a: np.sum(a.code_bytes_by_tile())
    EXCHANGE_CODE = "Exchange code", lambda a: np.sum(a.exchange_code_by_tile())
    CONTROL_CODE = "Control code", lambda a: np.sum(a.control_code_by_tile())
    VARIABLES = "Variables", lambda a: np.sum(a.max_vars_usage())

    def __new__(cls, *_args, **_kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, name, value_function):
        self.column_name = name
        self.value_function = value_function

    def as_string(self, name, layer):
        '''Get this field as a string.'''
        layer.name = name
        return str(self.value_function(layer))


class Field(Enum):
    '''List of columns in spreadsheet in order starting from A.
    Format used: ENTRY = "<Column name>", <function to get associated value>"'''

    LAYER = "Layer", lambda a: a.layer_name + str(a.layer_name_note)
    FULL_NAME = "Full name", lambda a: a.name
    VERTEX_STATE = "Vertex state", lambda a: a.operation.vertex_state_bytes()
    VERTEX_CODE = "Vertex code", lambda a: a.operation.vertex_code_bytes()
    EXCHANGE_CODE = "Exchange code", lambda a: a.operation.exchange_code()
    CONTROL_CODE = "Control code", lambda a: a.operation.control_code_bytes()
    VARIABLES = "Variables", lambda a: a.operation.variable_bytes()

    def __new__(cls, *_args, **_kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, name, value_function):
        self.column_name = name
        self.column_letter = chr(self.value + ord('A') - 1)
        self.value_function = value_function

    def as_string(self, operations, op_index):
        '''Get this field as a string.'''
        return str(self.value_function(operations[op_index]))
