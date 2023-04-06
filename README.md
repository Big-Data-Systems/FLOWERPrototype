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

To run, use

```
python src/flower.py path
```

Where `path` is an absolute or relative path (supporting glob syntax) to one or more files.

Flags:
```
-v          Verbose mode: print every state node visited and its ancestors.
-o path     Output: write the analysis (a json file), or otherwise print output.
-e path     ER Output: write the FLOWER diagram json to the path.
-d          Derived: Include derived/intermediate entities from flow nodes in the diagram json.
```