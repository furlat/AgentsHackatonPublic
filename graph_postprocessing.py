from load_annotations import Graph, Node, Action, character_splitter
from typing import List, Dict

STOP_CHARACTERS = [
        "he", "she", "it", "they", "someone", "anyone", "everybody", "anybody",
        "person", "thing", "creature", "man", "woman", "boy", "girl", "people",
        "nobody", "nothing", "narrator", "herself", "himself", "hers", "him",
        "themselves", "us", "we", "you", "yourself", "yourselves", "i", "me", "myself",
        "it", "themselves", "him", "her", "hers", "himself", "herself", "itself"
    ]

def getting_probable_entities(nodes: List[Node], valid_edges: List[Action]):
    """
    Naive function to filter out stop characters and non-probable entities.
    """
    filtered_nodes = [n for n in nodes if n.node_name.strip().lower() not in STOP_CHARACTERS and n.node_name.istitle()]
    names = {n.node_name for n in filtered_nodes}
    valid_edges = [e for e in valid_edges if (e.source in names) and (e.target in names)]
    nodes = filtered_nodes
    return nodes, valid_edges

def split_nodes(nodes: List[Node]) -> List[Node]:
    """
    Given a list of Node objects, split any composite nodes (is_single=False) into individual parts,
    and merge duplicates by node_name (unioning action participation lists).
    """
    node_map: Dict[str, Node] = {}
    for node in nodes:
        if not node.is_single:
            parts = character_splitter.split(node.node_name)
            for part in parts:
                name = part.strip()
                if not name:
                    continue
                existing = node_map.get(name)
                if existing is None:
                    # create new single node from composite
                    new_node = Node(
                        node_name=name,
                        is_single=True,
                        is_character=node.is_character,
                        partakes_as_source_in_actions=list(node.partakes_as_source_in_actions),
                        partakes_as_target_in_actions=list(node.partakes_as_target_in_actions),
                        entities_in_node_name=[name]
                    )
                    node_map[name] = new_node
                else:
                    # merge metadata
                    # is_character
                    if node.is_character:
                        object.__setattr__(existing, 'is_character', True)
                    # merge participation lists
                    merged_src = set(existing.partakes_as_source_in_actions) | set(node.partakes_as_source_in_actions)
                    merged_tgt = set(existing.partakes_as_target_in_actions) | set(node.partakes_as_target_in_actions)
                    object.__setattr__(existing, 'partakes_as_source_in_actions', list(merged_src))
                    object.__setattr__(existing, 'partakes_as_target_in_actions', list(merged_tgt))
        else:
            # keep single nodes unchanged, but merge if duplicate appears
            existing = node_map.get(node.node_name)
            if existing is None:
                node_map[node.node_name] = node
            else:
                # merge metadata
                if node.is_character:
                    object.__setattr__(existing, 'is_character', True)
                merged_src = set(existing.partakes_as_source_in_actions) | set(node.partakes_as_source_in_actions)
                merged_tgt = set(existing.partakes_as_target_in_actions) | set(node.partakes_as_target_in_actions)
                object.__setattr__(existing, 'partakes_as_source_in_actions', list(merged_src))
                object.__setattr__(existing, 'partakes_as_target_in_actions', list(merged_tgt))
    # return sorted for consistent ordering
    return sorted(node_map.values(), key=lambda n: n.node_name)


def character_subgraph(graph: Graph, split: bool = False, naive_beautify: bool = False) -> Graph:
    """
    Return a new Graph containing:
      - Nodes where `is_character` is True
      - Edges where either source or target is in those nodes
    If split=True, composite nodes will be split into individual single nodes.
    """
    # filter character nodes
    char_nodes = [n for n in graph.nodes if n.is_character]
    char_names = {n.node_name for n in char_nodes}
    # filter edges
    valid_edges = [edge for edge in graph.edges if (edge.source in char_names) or (edge.target in char_names)]
    # optionally split composite nodes
    nodes = split_nodes(char_nodes) if split else char_nodes
    nodes, valid_edges = getting_probable_entities(nodes, valid_edges) if naive_beautify else (nodes, valid_edges)
    return Graph(nodes=nodes, edges=valid_edges)


def character_interaction_graph(graph: Graph,
                                split: bool = False,
                                naive_beautify: bool = False) -> Graph:
    """
    Return a new Graph containing:
      - Nodes where `is_character` is True
      - Edges where either source or target is in those nodes
    If split=True, composite nodes will be split into individual single nodes.
    Note: self loops are discarded.
    """
    # 1) pick only character-to-character edges, drop self-loops
    edges = [
        a for a in graph.edges
        if a.source_is_character
           and a.target_is_character
           and a.source != a.target
    ]

    # 2) determine surviving node names
    node_names = {e.source for e in edges} | {e.target for e in edges}
    # pull in the Node objects
    nodes = [n for n in graph.nodes if n.node_name in node_names]

    # 3) optional split and/or beautify
    if split:
        nodes = split_nodes(nodes)
    

    # 4) rebuild participation lists on the final nodes
    node_map = {n.node_name: n for n in nodes}
    for n in node_map.values():
        object.__setattr__(n, 'partakes_as_source_in_actions', [])
        object.__setattr__(n, 'partakes_as_target_in_actions', [])
    for action in edges:
        if action.source in node_map:
            node_map[action.source].partakes_as_source_in_actions.append(action.action_id)
        if action.target in node_map:
            node_map[action.target].partakes_as_target_in_actions.append(action.action_id)
    
    nodes = list(node_map.values())
    nodes, edges = getting_probable_entities(nodes, edges) if naive_beautify else (nodes, edges)

    return Graph(nodes=nodes, edges=edges)
    
