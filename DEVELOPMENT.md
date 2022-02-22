## Design

The overall design of PopLiner is as follows:

 - Data is loaded from the Poplar profile and cached in order to accelerate
   later computation.
 - All of the programs in the graph are identified by walking over the program
   tree.  A poplar program can represent a number of different actions carried
   out on the IPU, for example executing a vertex on a tile or synchronising
   the tiles on the IPU.
 - Each program is associated with a high-level operation by looking at its
   debug contexts.  A list of distinct operations is collected, each
   operation containing one or more programs.
 - Data is collected on the memory requirements for each operation.
 - We now have the information needed to efficiently split the operations
   across multiple IPUs.  PopLiner writes this information to a spreadsheet
   for the user to manually decide on the splits.
 - Optionally, PopLiner can run a solver to automatically split the operations
   across a given number of IPUs.

## Repository layout

The body of the PopLiner code can be found in the `popliner` directory, while
tests are located in the `tests` directory.

#### [popliner/main.py](popliner/main.py)

The entry point to the PopLiner code is `popliner/main.py`.  This code handles
command-line arguments and the high-level program flow.  After arguments have
been parsed, the profile report is loaded using the `openReport` function from
libpva.  The steps in the report are grouped into a sequence of high-level
operations (which are optionally written to the output file).

For the CSV and TSV output formats, a summary of memory usage is printed and,
optionally, layer affinity (the size of shared variables between each layer
pair) and inter-layer communication (the size of variables created in a
layer and consumed in the following layer).

If the `--solve` argument is passed, then PopLiner will attempt to
automatically split up the model to fit on the number of IPUs specified in the
`--num-ipus` argument.  The details of the splits are then written to the
output.

#### [popliner/operation.py](popliner/operation.py)

This file contains the `Operation` class which represents high-level
operations in the graph.  Each operation contains one or more programs from
the overall program tree.  The class provides methods which give information
about the memory consumption of the programs in the operation.

This file also contains the `Program` class which represents a single Poplar
program.

#### [popliner/operation_list.py](popliner/operation_list.py)

This file contains two classes, `NamedOperation` and `OperationList`.

`NamedOperation` is a wrapper around an operation which associates a name and
type with the operation.  Note that this class stores the UID of the operation
and does not store an actual `Operation` object.  This is an optimisation: some
models re-use the same operation many times with different names so using
NamedOperation minimises the additional cost of each name.

`OperationList` stores a list of operations.  The basic data structure is an
ordered list of `NamedOperation` objects and a mapping from each operation
UID to a corresponding `Operation` object.  The list is populated by walking
the report's program tree and debug context layers.  The program tree is
walked in order to collect a list of all of the programs in the report.  Then,
for each program found, we inspect all of the debug context layers it is
contained within as these provide information about which high-level operation
a program is part of.  The operations found from the debug context information
are all added to the list of operations stored in `OperationList`.

#### [popliner/spreadsheet.py](popliner/spreadsheet.py)

This file contains classes used to represent spreadsheet cell formulae.  When
the tab-separated values (TSV) output format is selected the output contains
formulae instead of just the final calculated values (this is helpful to
understand where values come from and in case the user wants to modify them).
`Formula` is the base class for spreadsheet formulae and provides methods
which return the formula text (for TSV output) and the final numerical value
calculated by evaluating the formula (for CSV output).

A number of other classes derive from `Formula` and represent the different
types of spreadsheet formulae used.

Finally, the `Field` class represents a single field in a spreadsheet (CSV or
TSV) as well as containing the list of columns in the spreadsheet.  It
contains lambda functions to get the value of the field depending which column
it is in.

#### [popliner/stage.py](popliner/stage.py)

This file contains the `Stage` class which represents a single stage in the
pipeline created by PopLiner's solver.  Each `Stage` holds a number of
`Operation` objects, representing the operations which PopLiner proposes to
place in that pipeline stage.  The `Stage` objects have a number of methods
which return information about the memory consumption on each tile of that
pipeline stage -- this is used to determine whether a stage has room for more
operations and whether the stage will fit on an IPU.

The `Stage` class also contains a number of static variables.  These cache
information loaded from the profile report to avoid needing to read data from
the report repeatedly.  The static variables are populated by the
`Stage.perform_static_analysis` static method.

#### [popliner/greedy_solver.py](popliner/greedy_solver.py)

This file contains the `GreedySolver` class which implements a greedy
algorithm to split an operation list to fit on multiple IPUs.  A greedy
algorithm is one which makes a series of locally optimal choices in the hope
of finding a good solution to a problem but which does not guarantee that the
solution will be the best possible (globally optimal) solution.

The main entry point into the solver is the `solve` method, which attempts to
split the operations onto the given number of IPUs.

The greedy algorithm used is effectively a bin-packing algorithm: the solver
tries to fit as many operations in the first IPU before moving to the second
IPU, etc.  This will tend to leave the final IPU or IPUs with less work to do.
Since an equal distribution of work between IPUs is desirable, a simple
technique is used to get a more even distribution of work:  we initially try
to only fill 60% of the memory on each IPU.  If this doesn't work because we
don't have enough memory, then we try to fill 70% of the memory on each IPU.
We follow this pattern until the target of 100% of the memory on each IPU.  If
this final attempt fails, then `num_ipus` must be increased.  Assuming that
`num_ipus` is not greater than double the required number this will give an
approximately equal (in terms of memory usage) division of operations across
the IPUs.

The solver also provides the `calculate_memory_affinity` and
`calculate_interlayer_exchange` methods.  These are not used by `solve` to
work out how to distribute operations over the available IPUs but provide
information which may help the user to see which layer boundaries would be
good candidates to split the graph.
