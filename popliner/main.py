# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

'''
Process a profile report from poplar (profile.pop and debug.cbor files) and
produce a spreadsheet containing information to help choose pipelining
split-points.  Can also optionally automatically solve for split points to fit
on a given number of IPUs.
'''

import logging
import json
import sys
import coloredlogs
from pva import openReport  # pylint: disable=no-name-in-module
from popliner.operation_list import OperationList
from popliner.greedy_solver import GreedySolver
import popliner.parse_args

VERSION = "1.0.0"

# import cProfile
# pr = cProfile.Profile()
# pr.enable()

LOGO = r"""
y.        r__    c ____                 _      _                     r__
0  *  y'   r\ \   c|  _ \   ___   _ __  | |    (_) _ __    ___  _ __  r\ \
y-     0*   r\ \  c| |_) | / _ \ | '_ \ | |    | || '_ \  / _ \| '__|  r\ \
0 '  y~    0` r) ) c|  __/ | (_) || |_) || |___ | || | | ||  __/| |      r) )
y~    0-  y' r/ /  c|_|     \___/ | .__/ |_____||_||_| |_| \___||_|     r/ /
0  *   y~  0r/_/  0-------------- c|_| 0-------------------------------  r/_/
y`0                                                         """
LOGO = LOGO.replace("0", '\33[0;1m')   # Reset colour
LOGO = LOGO.replace("r", '\33[31;1m')  # Red
LOGO = LOGO.replace("c", '\33[96;1m')  # Cyan
LOGO = LOGO.replace("y", '\33[33;1m')  # Yellow
print(LOGO + '\u001b[0mv' + VERSION + "\n", file=sys.stderr)

logger = logging.getLogger("root")
coloredlogs.install(level='INFO', fmt='%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')

args = popliner.parse_args.get_args()

if args.load_from_file:
    assert args.save_to_file is None, "Cannot use --save-to-file with --load-from-file."
    assert args.debug is None, "Do not provide 'debug' path with --load-from-file."
    operations = OperationList.from_file(args.profile, args)
else:
    assert args.debug, "Please provide a 'debug' path."
    logger.info("Loading report...")
    report = openReport(args.profile, args.debug)
    operations = OperationList(report, args)

solver = GreedySolver(operations)
DELIMITER = ',' if args.format == "csv" else '\t'

if args.operation_breakdown:
    print(operations.as_csv(delimiter=DELIMITER))

if args.layer_breakdown:
    print(operations.layers_as_csv(delimiter=DELIMITER))

if args.memory_totals:
    total = solver.get_memory_for_layers()
    print("Total memory: " + str(sum(total["total_mem"])))

    TILES_TO_SHOW = 10
    # Compare total memory by category
    print("Category" + DELIMITER, end='')
    for i in range(TILES_TO_SHOW):
        print("Tile " + str(i) + DELIMITER, end='')
    print("")

    if args.profile:
        print("Always live", end='')
        for i in range(TILES_TO_SHOW):
            print(DELIMITER + str(report.compilation.tiles[i].memory.alwaysLiveBytes), end='')
        print("")
        print("Not-always live", end='')
        for i in range(TILES_TO_SHOW):
            print(DELIMITER + str(report.compilation.tiles[i].memory.notAlwaysLiveBytes), end='')
        print("")
    else:
        print("Provide profile.pop path to see variable memory usage by tile.")

if args.memory_affinity:
    solver.calculate_memory_affinity()

if args.interlayer_communication:
    solver.calculate_interlayer_exchange()

if args.solve:
    success_mem_prop = solver.solve(args.num_ipus, args.mem_per_tile)
    if success_mem_prop:
        splits = solver.get_splits_totals()
        print(json.dumps(splits, indent=4))
        layers = operations.layers()
        for split in splits[:-1]:
            layers.insert(layers.index(split["layer_to"])+1, "|")
        logger.info("SUCCESS: %s", " ".join(["["] + [str(layer) for layer in layers] + ["]"]))
        if len(splits) != args.num_ipus:
            logger.warning("Used %d IPUs instead of the %d requested.", len(splits), args.num_ipus)
    else:
        splits = solver.get_splits_totals()
        print(json.dumps(splits, indent=4))

        logger.info("FAILURE: Unable to fit model on IPUs. Best effort printed.")


# pr.disable()
# pr.print_stats(sort='time')
