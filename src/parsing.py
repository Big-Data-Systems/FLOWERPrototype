import ast
from dataclasses import dataclass
import functools
from typing import Optional, Tuple
from uuid import uuid4 as uuid

"""
Flow analysis module.

Main object is the `Flow`, for which a source AST is given and a list of nodes is written out.
"""

# Basic example of the concept of "extensions" to a FLOWER tool:
# We define a set of patterns (which in our case is simply a number of known functions) that
# we consider to be "reads" or "writes", while everything else is a stateful operation if
# it acts on a state.

read_patterns = ("read_csv", "read_pickle", "read_table", "read_json")
write_patterns = ("to_csv", "to_json")



""" flow algorithm
for line in flow:
    if read:
        state = read()
        flow.addread(state)
    elif write:
        flow.addwrite(state)

    if state created:
        flow.add(state)
    elif state modified from [states]:
        flow[state] = state.newfrom(...states)
"""



class FlowVisitor(ast.NodeVisitor):
    """
    Visits any "scope"/executed list of instructions to search for changes in state
    """

    def __init__(self, flow: "Flow") -> None:
        super().__init__()
        self.flow = flow

    # assignment statements (any x (=) y type statement) will update
    # a state within a flow using some expression.
    def visit_AnnAssign(self, node: ast.AnnAssign):
        self.flow.update_state(node.target, node.value)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            match target:
                case ast.Tuple() | ast.List():
                    for el in target.elts:
                        self.flow.update_state(el, node.value)
                case _:
                    self.flow.update_state(target, node.value)

    def visit_AugAssign(self, node: ast.AugAssign):
        self.flow.update_state(node.target, node.value, node.op)

    def visit_Call(self, node: ast.Call):
        self.flow.write_state(node)

    # ignore control flow statements as if they are not there
    # for more advanced use, we can consider these as many to many or
    # conditional input values in pipeline inference.
    def visit_If(self, node: ast.If):
        self.visit(node.test)
        for n in node.body:
            self.visit(n)
        for n in node.orelse:
            self.visit(n)

    def visit_For(self, node: ast.For):
        self.flow.update_state(node.target, node.iter)
        for n in node.body:
            self.visit(n)

    def visit_While(self, node: ast.While):
        self.visit(node.test)
        for n in node.body:
            self.visit(n)
        for n in node.orelse:
            self.visit(n)


class StateVisitor(ast.NodeVisitor):
    """
    Visits expression nodes searching for nested states of a given flow
    """

    def __init__(self, flow: "Flow") -> None:
        super().__init__()
        self.flow = flow

    def find_states(self, node: ast.AST) -> tuple["State"]:
        if names := self.visit(node):
            nameset = set(names)
            return tuple(v for k, v in self.flow.states.items() if k in nameset)
        return []

    def generic_visit(self, _):
        return []

    def visit_UnaryOp(self, node: ast.UnaryOp):
        return self.visit(node.operand)

    def visit_BinOp(self, node: ast.BinOp):
        return self.visit(node.left) + self.visit(node.right)

    def visit_BoolOp(self, node: ast.BoolOp):
        return [v for vs in node.values for v in self.visit(vs)]

    def visit_Call(self, node: ast.Call):
        return (
            self.visit(node.func)  # potentially a chain of calls
            + [v for vs in node.args for v in self.visit(vs)]
            + [v for vs in node.keywords for v in self.visit(vs.value)]
        )

    def visit_Name(self, node: ast.Name):
        return [base_name(node)]

    def visit_Attribute(self, node: ast.Attribute):
        # discard attribute name, assuming it is not relevant to the base state.
        # for stateful analysis, this should instead be looked at as a possible key.
        return self.visit(node.value)

    def visit_Subscript(self, node: ast.Subscript):
        return self.visit(node.value) + self.visit(node.slice)

    def visit_Slice(self, node: ast.Slice):
        return self.visit(node.lower) + self.visit(node.upper) + self.visit(node.step)

    def visit_Index(self, node: ast.Index):
        return self.visit(node.value)

    def _visit_iterable(self, node):
        return functools.reduce(lambda l, el: l + self.visit(el), node.elts, [])

    def visit_List(self, node):
        return self._visit_iterable(node)
    def visit_Tuple(self, node):
        return self._visit_iterable(node)
    def visit_Set(self, node):
        return self._visit_iterable(node)


@dataclass
class StateOp:
    name: str
    args: Tuple[str]

    def __str__(self) -> str:
        return f"{self.name}: [{self.args}]"


class State:
    def __init__(self, name: str, op: StateOp, ancestors=None, descendants=None):
        self.op: StateOp = op
        self.name: str = name
        self.ancestors: list[State] = list(ancestors or [])
        self.descendants: list[State] = list(descendants or [])
        self.writes = []
        self.reads = []
        self.uuid = str(uuid())

        if op.name == "read":
            self.reads.append(op.args)

    def add_write(self, resource):
        self.writes.append(resource)

    def summarize(self):
        return {
            "id": self.uuid,
            "name": self.name,
            "ancestors": [anc.uuid for anc in self.nearest_interesting_ancestors()],
            "descendants": [des.uuid for des in self.nearest_interesting_descendants()],
            "ops": [str(op) for op in self.summarize_ops()],
            "writes": [str(w) for w in self.writes],
            "reads": [str(r) for r in self.reads],
        }

    def interesting(self) -> bool:
        # decide if this node is "interesting" for analysis purposes, meaning one of:
        #   - multiple ancestors (result of a merge)
        #   - multiple descendants (copied into separate states)
        #   - state is read from or written to an external state
        #   - has a child that is a result of a merge
        return (
            len(self.writes) > 0
            or len(self.ancestors) != 1
            or len(self.descendants) > 1
            or any(len(d.ancestors) > 1 for d in self.descendants)
        )

    def nearest_interesting_ancestors(self):
        if len(self.ancestors) < 1:
            return []
        elif len(self.ancestors) == 1 and not self.ancestors[0].interesting():
            return self.ancestors[0].nearest_interesting_ancestors()
        else:
            return self.ancestors

    def nearest_interesting_descendants(self):
        if len(self.descendants) < 1:
            return []
        elif len(self.descendants) == 1 and not self.descendants[0].interesting():
            return self.descendants[0].nearest_interesting_descendants()
        else:
            return self.descendants

    def report(self, l=0):
        print(
            self.name
            + " <- "
            + str(self.op)
            + (f"({len(self.writes)} writes)" if len(self.writes) else "")
        )
        for anc in self.ancestors:
            print(" " * l + anc.name + " <- " + str(anc.op))
            anc.report(l + 1)

    def summarize_ops(self) -> list[StateOp]:
        if len(self.ancestors) == 1 and not self.ancestors[0].interesting():
            return self.ancestors[0].summarize_ops() + [self.op]
        else:
            return [self.op]

    def compile_lineage(self) -> dict:
        out = {self.uuid: self.summarize()}
        for anc in self.ancestors:
            out.update(anc.compile_lineage())
        return out

    def get_inputs(self) -> list[str]:
        if len(self.ancestors) < 1:
            return [self.op]
        else:
            return [inp for anc in self.ancestors for inp in anc.get_inputs()]


def interpret_v(node: ast.AST):
    # Interpret a value to a human-readable expression, if possible.
    
    match node:
        case ast.Constant(value=v):
            return str(v)
        case ast.Name(id=v):
            return f"variable: {v}"
    return node


def atomic_name(node: ast.AST) -> str:
    # gets the smallest unit of an Attribute or Name node's name, such as
    # obj.func(args) -> "func"  or  func(args) -> func
    match node:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name
        case ast.Subscript(value=nested):
            return atomic_name(nested) + "[]"
        case ast.Call(func=nested):
            return atomic_name(nested)
        case ast.BinOp(op=op) | ast.UnaryOp(op=op):
            return op.__class__.__name__
        case ast.List():
            return "list()"
        case ast.Tuple():
            return "tuple()"
        case ast.Set():
            return "set()"
    raise TypeError(f"Node does not appear to have an atomic name: {node}")


def base_name(node: ast.AST) -> str:
    # gets the base unit of a node's name, such as
    # name.a.b.c -> "name"   or   name[a].b.c -> "name"   or   name -> name
    match node:
        # basic use: get root name and consider that the state key
        case ast.Name(id=name):
            return name
        case ast.Attribute(value=nested) | ast.Subscript(value=nested) | ast.Call(
            func=nested
        ):
            return base_name(nested)
    raise TypeError(f"Node does not appear to have a base name: {node}")




def get_read_key(source: ast.AST) -> Optional[str]:
    # identify whether AST node is considered a "read" by this program.
    # if true, returns a resource key indicating the resource being read from.
    match source:
        case ast.Call():
            if atomic_name(source) in read_patterns:
                return interpret_v(source.args[0])
    return None


def get_write_key(source: ast.AST) -> Optional[str]:
    # identify whether AST node is considered a "write" by this program.
    # if true, returns a resource key indicating the resource being written to.
    match source:
        case ast.Call():
            if atomic_name(source) in write_patterns:
                return interpret_v(source.args[0])
    return None


class Flow:
    def __init__(self, source: ast.AST, verbose=False):
        self.source = source
        self.states: dict[State] = {}
        self.verbose = verbose

        # walk through source and try to build tree
        visitor = FlowVisitor(self)
        visitor.visit(source)

        if verbose:
            for s in self.states.values():
                s.report()

    # parse values of an assignment expression to update a state in this flow.
    # operates with expressions of the form target = source
    def update_state(
        self, target: ast.AST, source: ast.AST, op: Optional[ast.AST] = None
    ):
        # options:
        # newstate <- state
        # newstate <- none

        key = self.get_key(target)

        if resource := get_read_key(source):
            # create a new state
            # node is of form key = x.read_y(resource)
            self.states[key] = State(key, StateOp("read", resource))
        elif source_states := self.get_source_states(source):
            # new state is being created from 1 or more old ones.
            # we call this a "transform" and name it after the function or atomic name
            # this states comes from.
            self.states[key] = State(
                key, StateOp("transform", atomic_name(source)), ancestors=source_states
            )
            for parent in source_states:
                parent.descendants.append(self.states[key])

    # parse an AST node to see if it is a writing node, updating state if so.
    def write_state(self, node: ast.AST):
        if resource := get_write_key(node):
            key = base_name(node)
            try:
                self.states[key].add_write(resource)
            except KeyError:
                if self.verbose:
                    print(
                        f'Skipping write for state "{key}" as it had no corresponding input file.'
                    )

    def get_key(self, target: ast.AST) -> str:
        # Converts a target AST to a state key (whether extant or not) in
        # the current Flow.
        # basic use: get root name and consider that the state key
        return base_name(target)

        # # for more in-depth analysis:
        # match:
        #     case ast.Attribute():
        #         return self.get_key(target.value) + f".{target.attr}"
        #     case ast.Subscript():
        #         return self.get_key(target.value) + "[]"
        # raise TypeError(f"target does not appear to be valid for assignment: {target}")

    def get_source_states(self, source: ast.AST) -> list[State]:
        # parse source AST (right side of assignment expression) for
        # keys of states we are currently keeping track of.
        return StateVisitor(self).find_states(source)

    def write_out(self):
        candidates = {}
        inputs = []
        outputs = []
        for output_state in (w for w in self.states.values() if len(w.writes) > 0):
            candidates.update(output_state.compile_lineage())
            inputs += [op.args for op in output_state.get_inputs()]
            outputs += output_state.writes

        return {
            "inputs": list(set(inputs)),
            "outputs": list(set(outputs)),
            "nodes": candidates,
        }