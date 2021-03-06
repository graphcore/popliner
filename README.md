# PopLiner

> Copyright 2021 Graphcore Ltd.

PopLiner is a command line tool and library used to determine optimal split points to use when
pipelining a large model.

This software is licensed under the MIT license, see LICENSE.txt for more details.

### Setup

PopLiner requires Python 3.6 or newer.  It depends on a few Python packages which can be installed
using `pip`:

```cmd
pip3 install -r requirements.txt
```

Before using PopLiner you must have an active Poplar installation.  See the
[Poplar SDK documentation](https://docs.graphcore.ai/projects/ipu-pod-getting-started/en/latest/installation.html#setting-up-the-sdk-environment)
for more details.  Poplar SDK version 2.5 or higher is required.  You can
activate the poplar installation in the `bash` shell as follows:

```cmd
source /path/to/poplar/install/enable.sh
```

You will need to reactivate the poplar installation each time you open a new terminal window.

### Running PopLiner

PopLiner can be run as follows, where `PROFILE_PATH` is the path to the profile generated by
poplar.  Output will be written to the `out.tsv` file.

```cmd
./popliner.py PROFILE_PATH/profile.pop PROFILE_PATH/debug.cbor >out.tsv
```

The inputs to PopLiner are a `profile.pop` and a `debug.cbor` file.  These are created by the
profiling functionality built in to Poplar.  By default PopLiner outputs tab-separated output
containing spreadsheet formulae.

PopLiner has the following options:

 - `-h`, `--help` Show a help message listing the available options
 - `--format {tsv,csv,json}` Select the format of the output
   - `tsv` Tab-separated values with spreadsheet formulae
   - `csv` Comma-separated values without spreadsheet formulae
   - `json` JSON format, intended to be consumed by other scripts or tools.  Only produced if `--solve` is used.
 - `--memory_totals` Print total memory by category.
 - `--solve` Solve for split points (beta).
 - `--num-ipus NUM_IPUS` When solving, the number of IPUs in the system (default=4).
 - `--mem_per_tile MEM_PER_TILE` When solving, the memory per tile in bytes (default=624kB).
 - `--operation_breakdown` Outputs a memory breakdown per operation.
 - `--layer_breakdown` Outputs a memory breakdown per layer.
 - `--memory_affinity` For each layer pair, outputs the size of shared variables.
 - `--interlayer_communication` For each layer pair, outputs the size of variables
   created in the first layer and consumed in the second one.

### Running the tests

After a poplar installation is activated and Python dependencies have been installed, tests can
be run using pytest:

```cmd
pytest
```

### Diagnostic tools

Popliner may suggest non-optimal or unfeasible splitting points if it is unable to correctly
interpret memory usage. This can be due to missing programs and variables that are the result
of missing debug context information among other possible reasons. The user can get insights
into the Popliner memory analysis by using some provided tools.

Script `tests/test_single_stage.py` analyses the differences between Popliner and the ground
truth provided by libpva. Among other checks, this script detects the execution bottleneck,
that is, the tile and step with the highest memory requirements. The script prints the list
of variables that are missing at that point. Note that the list may contain false positives,
so a visual inspection by the user may be required to detect the culprit of the discrepancy.
A recommended starting point is to identify the largest of the missing variables and to use
Popvision Graph Analyser to investigate the possible causes of such an omission.

### Development

[DEVELOPMENT.md](DEVELOPMENT.md) contains information about the structure of the repository and
how PopLiner works.  It is intended as an introduction for people who would like to modify the
code or base other work upon it.
