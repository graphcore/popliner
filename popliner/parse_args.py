# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

'''
Parser for all PopLiner command line arguments.
'''

import argparse

DEFAULT_LAYER_NAME_REGEX = \
    r"(?:^|\/)(?:[L|l]ayer|blocks|encoder)[\/_\.]?(\d+)"
DEFAULT_INPUTS_REGEX_LAYER_0 = r"(word_embeddings|Embedding_Dict)"


def _get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Process profile.pop and debug.cbor files of a large
model compiled for a single IPU (out-of-memory) and provide guidance on how to split the model
across multiple IPUs using pipelining.""",
        epilog=f"""NOTE: When generating profile files for PopLiner, the environment variable at the
end of this text should be set. The same value should be used for --layer-name-regex. Remember to
unset this variable when compiling with the suggested split points:
POPART_POPLINER_OUTLINER_REGEX='{DEFAULT_LAYER_NAME_REGEX}'""")

    parser.add_argument('--format', choices=('tsv', 'csv'), default='tsv',
                        help='''Select tab-separated (tsv) or comma-separated (csv) output format
for --operation-breakdown, --layer-breakdown and --memory-totals.''')
    parser.add_argument('--memory-totals', action='store_true',
                        help='Print total memory by category.')
    parser.add_argument('--solve', action='store_true', help='Solve for split points.')
    parser.add_argument('--layer-order', choices=('natural', 'steps'), default='steps',
                        help='''Sort layers by name (natural) or by program steps (steps).''')
    parser.add_argument('--num-ipus', type=int, default=16,
                        help='When solving, the maximum number of IPUs in the system.')
    parser.add_argument('--mem-per-tile', type=int, default=638_976,
                        help='When solving, the memory per tile in bytes.')
    parser.add_argument('--operation-breakdown', action='store_true',
                        help='Outputs a memory breakdown per operation.')
    parser.add_argument('--layer-breakdown', action='store_true',
                        help='Outputs a memory breakdown per layer.')
    parser.add_argument('--memory-affinity', action='store_true',
                        help='For each layer pair, outputs the size of shared variables.')
    parser.add_argument('--interlayer-communication', action='store_true',
                        help='''For each layer pair, outputs the size of variables created in the
                                first layer and consumed in the second one.''')
    parser.add_argument('--save-to-file', help='Pre-process and save operations to this file path.')
    parser.add_argument('--load-from-file', action='store_true',
                        help='Load pre-processed operations from this file path.')
    parser.add_argument('--layer-name-regex', default=DEFAULT_LAYER_NAME_REGEX,
                        help='''Regular expression used to extract layer name (in capture group)
                                from operation names.''')
    parser.add_argument('--inputs-regex-layer-0', default=DEFAULT_INPUTS_REGEX_LAYER_0,
                        help='''If this regular expression matches the name of any input variable of
                                an operation, the operation will be assigned to the first layer.''')
    parser.add_argument('profile', help='Path to profile.pop file.')
    parser.add_argument('debug', help='Path to debug.cbor file.', nargs='?', default=None)
    return parser


def get_args(args=None):
    '''If args=None, get arguments from command line. Otherwise get arguments from list provided as
    args (the value for "profile" will be empty).'''
    if args is not None:
        # If overriding args, insert blank value for profile as it is only used by main.py
        args = [""] + args
    return _get_parser().parse_args(args)


def default_args():
    '''Get all default values. The value for "profile" will be empty.'''
    return _get_parser().parse_args([""])
