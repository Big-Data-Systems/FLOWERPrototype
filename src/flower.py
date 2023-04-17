from typing import Iterable


"""
Important to note that this will only apply to static (non-interactive)
pipelines, so changing file names can generally not be inferred.

A more advanced tool would use a set of inference techniques
outside the scope of this prototype to better guess file names,
keys, attributes and relationships.
"""

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


def make_rel(
    frm: str,
    to: str,
    *,
    color="black",
    dash=[3, 2],
    arr="Standard",
    width="1",
    text="",
    toText="",
):
    return {
        "from": frm,
        "to": to,
        "color": color,
        "dash": dash,
        "arr": arr,
        "width": width,
        "text": text,
        "toText": toText,
    }


def gen_rels(flows: dict, detailed=False) -> list[dict]:
    rels = [
        make_rel(resource, name)
        for name, flow in flows.items()
        for resource in flow["inputs"]
    ] + [
        make_rel(name, resource)
        for name, flow in flows.items()
        for resource in flow["outputs"]
    ]
    if detailed:
        return (
            rels
            + [  # node -> node and node -> files relations
                make_rel(nid, to)
                for flow in flows.values()
                for nid, node in flow["nodes"].items()
                for k in ["descendants", "writes"]
                for to in node[k]
            ]
            + [  # file -> node relations
                make_rel(frm, nid)
                for flow in flows.values()
                for nid, node in flow["nodes"].items()
                for frm in node["reads"]
            ]
        )
    return rels


def make_ent(
    name: str,
    pks: Iterable[str] = [],
    fks: Iterable[str] = [],
    attrs: Iterable[str] = [],
    *,
    items=[],
    color=None,
    derived=False,
    **kw,
):
    color = color if color is not None else "#82E0AA" if derived else "#fff9ff"
    return {
        "key": name,
        "items": [
            {
                "name": k,
                "iskey": k in pks,
                "figure": "Decision",
                "color": "red" if k in pks else "green" if k in fks else "white",
            }
            for k in {*pks, *fks, *attrs}
        ]
        + items,
        "colorate": color,
        **kw,
    }


def gen_ents(flows: dict, detailed=False) -> list[dict]:
    ents = [make_ent(name) for name in flows] + [
        make_ent(resource)
        for resource in {
            r for node in flows.values() for k in ("inputs", "outputs") for r in node[k]
        }
    ]
    if detailed:
        return ents + [
            make_ent(
                nid,
                ops=node["ops"],
                parent_flow=fname,
                derived=True,
                node_name=node["name"],
            )
            for fname, flow in flows.items()
            for nid, node in flow["nodes"].items()
        ]
    return ents


