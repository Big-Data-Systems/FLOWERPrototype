### FLOWER: A prototype

This is a prototype for the upcoming paper on FLOWER (FLOW + ER): A technique for extending
Entity-Relationship diagrams. 

ER is useful but limited to static relationships: We cannot see how data moves or their
relationship with other data. By extending our view of ER to include the concept of "Flows"
between entities, we can better understand systems at a glance.

This prototype is meant for demonstrating a working implementation of the algorithm for 
analyzing code for use with FLOWER. Provided with a set of files, the program will parse
the top level semantic trees (non-functional, non class, current module only) and construct
an analysis of the inputs, outputs and relationships between these through the pipeline.

### Usage

This project uses python 3.10 -- no external libraries are required.

```
usage: flower.py [-h] [-v] [-o OUTPUT] [-e ERDIAG] [-d] filename

positional arguments:
  filename              Relative or absolute path to file(s) to parse. Supports glob.

options:
  -h, --help            show this help message and exit
  -v, --verbose
  -o OUTPUT, --output OUTPUT
                        Write the analysis (a json file). If unset, prints output directly.
  -e ERDIAG, --erdiag ERDIAG
                        Output FLOWER diagram json files. Will be appended with _ents.json and _rels.json.
  -d, --derived         Also include derived entities (internal state nodes like dataframes) in ER output.
```