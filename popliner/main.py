# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

'''
Process a profile report from poplar (profile.pop and debug.cbor files) and
produce a spreadsheet containing information to help choose pipelining
split-points.  Can also optionally automatically solve for split points to fit
on a given number of IPUs.
'''

import argparse
import logging
from pva import openReport  # pylint: disable=no-name-in-module
from popliner.operation_list import OperationList
from popliner.greedy_solver import GreedySolver

# import cProfile
# pr = cProfile.Profile()
# pr.enable()

formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger("root")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

FORMAT_HELP = """
Select the format of the output.
tsv: Tab-separated values.
csv: Comma-separated values.
json: JSON format. Only results in output if --solve is also used.
(default = tsv)
"""

parser = argparse.ArgumentParser(description='Process profile.pop and debug.cbor and produce a \
spreadsheet containing information to help decide pipelining split-points.')
parser.add_argument('--format', choices=('tsv', 'csv', 'json'), default='tsv', help=FORMAT_HELP)
parser.add_argument('--memory_totals', action='store_true', help='Print total memory by category.')
parser.add_argument('--solve', action='store_true', help='Solve for split points (beta).')
parser.add_argument('--num-ipus', type=int, default=4,
                    help='When solving, the number of IPUs in the system (default=4).')
parser.add_argument('--mem_per_tile', type=int, default=638976,
                    help='When solving, the memory per tile in bytes (default=624kB).')
parser.add_argument('--operation_breakdown', action='store_true',
                    help='Outputs a memory breakdown per operation.')
parser.add_argument('--layer_breakdown', action='store_true',
                    help='Outputs a memory breakdown per layer.')
parser.add_argument('--memory_affinity', action='store_true',
                    help='For each layer pair, outputs the size of shared variables.')
parser.add_argument('--interlayer_communication', action='store_true',
                    help='''For each layer pair, outputs the size of variables created in the
                          first layer and consumed in the second one.''')
parser.add_argument('profile', help='Path to profile.pop file.')
parser.add_argument('debug', help='Path to debug.cbor file.')
args = parser.parse_args()

logger.info("Loading profile...")
report = openReport(args.profile, args.debug)

operations = OperationList(report)
DELIMITER = ',' if args.format == "csv" else '\t'

if args.operation_breakdown:
    if args.format == 'csv':
        print(operations.as_csv(delimiter=DELIMITER))
    elif args.format == 'tsv':
        print(operations.as_csv(delimiter=DELIMITER))

if args.layer_breakdown:
    print(operations.layers_as_csv(delimiter=DELIMITER))

#######################################################
# Compare total memory by category
if args.memory_totals:
    solver = GreedySolver(report, operations)
    total = solver.get_single_stage_mem_for_inference()
    print("Total memory: " + str(sum(total["total_mem"])))

    TILES_TO_SHOW = 10
    print("Category" + DELIMITER, end='')
    for i in range(TILES_TO_SHOW):
        print("Tile " + str(i) + DELIMITER, end='')
    print("")

    print("Always live", end='')
    for i in range(TILES_TO_SHOW):
        print(DELIMITER + str(report.compilation.tiles[i].memory.alwaysLiveBytes), end='')
    print("")
    print("Not-always live", end='')
    for i in range(TILES_TO_SHOW):
        print(DELIMITER + str(report.compilation.tiles[i].memory.notAlwaysLiveBytes), end='')
    print("")

    if args.memory_affinity:
        solver.calculate_memory_affinity()
    if args.interlayer_communication:
        solver.calculate_interlayer_exchange()
#######################################################

if args.solve:
    solver = GreedySolver(report, operations)
    success = solver.solve(args.num_ipus, args.mem_per_tile)

    if args.format == 'json':
        print(solver.get_splits_as_json())
    else:
        print("")
        if not success:
            print("Unable to fit model in ipus")
        else:
            header = ["layer_from", "layer_to", "total_mem", "variables", "vertex_code",
                      "vertex_state", "exchange_code"]
            print(DELIMITER.join(header))
            for split in solver.get_splits_totals():
                my_list = [split["layer_from"],
                           str(split["layer_to"]),
                           str(split["mem"]["total_mem"]),
                           str(split["mem"]["variables"]),
                           str(split["mem"]["vertex_code"]),
                           str(split["mem"]["vertex_state"]),
                           str(split["mem"]["exchange_code"])]
                print(DELIMITER.join(my_list))

# pr.disable()
# pr.print_stats(sort='time')
