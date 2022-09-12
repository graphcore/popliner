# PopLiner

> Copyright 2021 Graphcore Ltd.
This software is licensed under the MIT license, see LICENSE.txt for more details.

PopLiner is a command line tool and library used to suggest split points to use when
pipelining a large model.  To do this, PopLiner analyses the out-of-memory profile files of a model
compiled for a single IPU.  Specifically, the required input files are a `profile.pop` and a
`debug.cbor` file, which are created by the profiling functionality built into Poplar.

The following command will output the PopLiner help text, which is referred to in this document:

```cmd
python3 ./popliner.py --help
```

### Preparing Your Model

PopLiner has been tested with PopART, PopTorch and TensorFlow models only.  Popliner currently does
not support the following:

  - Training in tensorflow (may work with some models),
  - Graph replication.

You must not use pipelining when generating the input profile for PopLiner.

PopLiner relies on explicit layer names being present in the model -- which must be added by the
user.  The way this is done depends on the framework used, but generally it is achieved by using
name scopes.  To ensure layer names have been correctly added, you may use PopVision Graph Analyser
to check that:

  - the `op_name` field in the `xla_op` debug context layers are populated for TensorFlow models;
  - the `opName` field in the `onnx` debug context layers are populated for PopART models;
  - the `op_name` field in the `poptorch` debug context layers are populated for PopTorch models.

PopLiner uses a regular expression to extract layers from these names - which can be customised
using a command line option.  By default, layer names are expected to start with "layer", "blocks"
or "encoder" then contain a number -- however, this is a simplified explanation and the exact
default regular expression can be seen in the PopLiner help text.

For PopART models (or any framework that uses PopART, like PopTorch), it is important to use the
`POPART_POPLINER_OUTLINER_REGEX` environment variable as explained in the PopLiner help text.  This
prevents PopART from over-optimising the model by combining components from all layers -- which is
not representative of how the model would be compiled for multiple IPUs.

Note: When applying the suggested split points from PopLiner, the same available memory proportion
must be used for all IPUs, and this value must be the same as that which was used to generate the
profile files provided to PopLiner.

### Setup

PopLiner requires Python 3.6 or newer.  It depends on a few Python packages which can be installed
using `pip`:

```cmd
python3 -m pip install -r requirements.txt
```

Before using PopLiner you must have an active Poplar installation.  See the
[Poplar SDK documentation](https://docs.graphcore.ai/projects/ipu-pod-getting-started/en/latest/installation.html#setting-up-the-sdk-environment) for more details.
Poplar SDK version 3.0 or higher is required.

### Running PopLiner

The basic command to run PopLiner is as follows:

```cmd
python3 ./popliner.py /path/to/profile.pop /path/to/debug.cbor
```

However, you will need to append command line options to this command to get useful output.  Please
see the PopLiner help text to understand the options.  Generally, you are most likely to want to use
both the `--solve` and `--num-ipus` flags.

### Running the tests

After a poplar installation is activated and Python dependencies have been installed, tests can
be run using pytest:

```cmd
pytest
```

### Diagnostic tools

Popliner may suggest non-optimal or unfeasible splitting points if it is unable to correctly
interpret memory usage.  This can be due to missing programs and variables that are the result
of missing debug context information among other possible reasons.  The user can get insights
into the Popliner memory analysis by using some provided tools.

Script `tests/test_single_stage.py` analyses the differences between Popliner and the ground
truth provided by libpva.  Among other checks, this script detects the execution bottleneck,
that is, the tile and step with the highest memory requirements.  The script prints the list
of variables that are missing at that point.  Note that the list may contain false positives,
so a visual inspection by the user may be required to detect the culprit of the discrepancy.
A recommended starting point is to identify the largest of the missing variables and to use
Popvision Graph Analyser to investigate the possible causes of such an omission.

### Development

[DEVELOPMENT.md](DEVELOPMENT.md) contains information about the structure of the repository and
how PopLiner works.  It is intended as an introduction for people who would like to modify the
code or base other work upon it.
