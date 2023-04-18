import argparse
import glob
import ast
import json
import time

from src.flower import gen_ents, gen_rels
from src.parsing import Flow


if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Write the analysis (a json file). If unset, prints output directly.",
    )
    parser.add_argument(
        "-e",
        "--erdiag",
        type=str,
        help="Output FLOWER diagram json files. Will be appended with _ents.json and _rels.json.",
    )
    parser.add_argument(
        "-d",
        "--derived",
        action="store_true",
        help="Also include derived entities (internal state nodes like dataframes) in ER output.",
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Relative or absolute path to file(s) to parse. Supports glob.",
    )
    args = parser.parse_args()

    flows = {}
    for fname in glob.glob(args.filename):
        if args.verbose:
            print(f"generating for {fname}")
        # ex: "data_sources/gamma.py"
        with open(fname) as f:
            tree = ast.parse(f.read())
        flo = Flow(tree, verbose=args.verbose)
        flows[fname] = flo.write_out()

    if args.output:
        fname = args.output if args.output.endswith(".json") else args.output + ".json"
        print(f"writing database info to {args.output}")
        with open(args.output, "w") as f:
            json.dump(flows, f, indent=2)
    else:
        print(json.dumps(flows, indent=2))

    if args.erdiag:
        for ending, func in (("_ents.json", gen_ents), ("_rels.json", gen_rels)):
            fname = args.erdiag + ending
            print(f"writing FLOWER info to {fname}")
            with open(fname, "w") as f:
                json.dump(func(flows, args.derived), f, indent=2)
    
    if args.verbose:
        print(f"Done in {time.time() - time_start} seconds.")
