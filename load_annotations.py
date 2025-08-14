from pydantic import BaseModel, Field
from tqdm import tqdm
from typing import List, Optional, Any, Union, Dict, ClassVar
import polars as pl
import re

PATH_TO_BOOKS = "./data/gutenberg_en_novels.parquet"


character_splitter = re.compile(r'\s*,\s*|\s*(?:and|\|)\s*', re.IGNORECASE)

all_books = pl.read_parquet(PATH_TO_BOOKS)

def clean_lines(lines):
    split_lines = lines.split("\n\r\n\r\n")
    newlines = []
    for line in split_lines: 
        l=re.sub('(\r\n)+\r?\n?',' ',line)
        l = re.sub(r"\s\s+"," ",l)
        if not re.search(r'^[^a-zA-Z0-9]+$',l):
            newlines.append(l.strip())
    return newlines

class Action(BaseModel):
    chunk_id: Union[int, str]
    chunk_text: str
    action_id: int
    action: str
    text_describing_the_action: str
    consequence: str
    text_describing_the_consequence: str
    source: Optional[str]
    source_type: Optional[str]
    source_is_character: bool
    target: Optional[str]
    target_type: Optional[str]
    target_is_character: bool
    location: Optional[str]
    temporal_order_id: int

    def __repr__(self) -> str:
        return f"<Action source={self.source} target={self.target}>"

    # Pydantic v2 config
    model_config = {"populate_by_name": True}


class Chunk(BaseModel):
    text_id: Union[int, str]
    text_had_no_actions: bool
    chunk_id: Union[int, str]
    chunk_text: str
    chunk_before: str
    chunk_after: str
    actions: List[Action] = []
    chunk_df: Optional[Any] = Field(default=None, exclude=True)

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.chunk_df is not None:
            self._build_actions(self.chunk_df)

    def _build_actions(self, chunk_df: pl.DataFrame) -> None:
        for row in chunk_df.iter_rows(named=True):
            act = Action(
                action_id=row['action_id'],
                action=row['action'],
                text_describing_the_action=row['text_describing_the_action'],
                consequence=row['consequence'],
                text_describing_the_consequence=row['text_describing_the_consequence'],
                source=row['source'].strip(),
                source_type=row['source_type'],
                source_is_character=row['source_is_character'],
                target=row['target'].strip(),
                target_type=row['target_type'],
                target_is_character=row['target_is_character'],
                location=row['location'],
                temporal_order_id=row['temporal_order_id'],
                chunk_id=self.chunk_id,
                chunk_text=self.chunk_text,
            )
            self.actions.append(act)

    def add_action(self, action: Action) -> None:
        self.actions.append(action)

    def __repr__(self) -> str:
        return f"<Chunk id={self.chunk_id} text_id={self.text_id} actions={len(self.actions)}>"

    model_config = {"populate_by_name": True}


class Node(BaseModel):
    node_name: str
    is_single: bool
    is_character: bool
    partakes_as_source_in_actions: List[Union[int, str]] = []
    partakes_as_target_in_actions: List[Union[int, str]] = []
    entities_in_node_name: List[str]

    def __repr__(self) -> str:
        return f"<Node name={self.node_name} single={self.is_single} char={self.is_character}>"

    def __hash__(self) -> int:
        return hash((self.node_name, self.is_single))

    model_config = {"populate_by_name": True, "frozen": True}



class Graph(BaseModel):
    nodes: List[Node]
    edges: List[Action]
    model_config = {"populate_by_name": True}

    def __repr__(self) -> str:
        return f"<Graph nodes={len(self.nodes)} edges={len(self.edges)}>"


class Book(BaseModel):
    book_id: Union[int, str]
    source: str = Field(..., alias='SOURCE')
    language: str
    gutenberg_id: Union[int, str]
    title: str
    issued: str
    authors: str
    subjects: str
    locc: Any
    bookshelves: Optional[str]
    chunks: List[Chunk] = []
    df: Optional[Any] = Field(default=None, exclude=True)

    model_config = {"populate_by_name": True}


    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.df is not None:
            self._build_chunks()

    def _build_chunks(self) -> None:
        unique_chunks = (
            self.df
            .select(["text_id", "chunk_id", "chunk", "text_had_no_actions"])
            .unique()
            .sort(["text_id", "chunk_id"])
        )

        grouped = unique_chunks.group_by("chunk_id")
        for _cid, chunk_rows in grouped:
            rows = list(chunk_rows.iter_rows(named=True))
            for i, row in enumerate(rows):
                before = "" if i == 0 else rows[i - 1]["chunk"]
                after = "" if i == len(rows) - 1 else rows[i + 1]["chunk"]
                sub_df = self.df.filter(pl.col("chunk_id") == row["chunk_id"])
                chunk = Chunk(
                    text_id=row["text_id"],
                    text_had_no_actions=row["text_had_no_actions"],
                    chunk_id=row["chunk_id"],
                    chunk_text=row["chunk"],
                    chunk_before=before,
                    chunk_after=after,
                    chunk_df=sub_df
                )
                self.chunks.append(chunk)

        self.chunks.sort(key=lambda c: c.chunk_id)

    def add_chunk(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)
    
    def get_book_text(self) -> str:
        """
        Return the full text of this book from a shared cached DataFrame.
        The book is split into paragraphs.
        """

        result = all_books.filter(pl.col("text_id") == self.gutenberg_id)

        if result.is_empty():
            raise ValueError(f"No text found for gutenberg_id: {self.gutenberg_id}")

        return clean_lines(result["TEXT"][0])

    def create_full_graph(self) -> Graph:
        """
        Build a Graph whose nodes are the unique sources/targets,
        each wrapped in a Node with detailed metadata. Edges are the Action instances.
        """
        node_info: Dict[str, Dict[str, Any]] = {}
        edges: List[Action] = []

        for chunk in self.chunks:
            for action in chunk.actions:
                edges.append(action)
                for side, is_source in ((action.source, True), (action.target, False)):
                    if not side:
                        continue
                    parts = character_splitter.split(side)
                    is_single = len(parts) == 1
                    name = side.strip()
                    info = node_info.setdefault(name, {
                        'is_single': is_single,
                        'is_character': False,
                        'partakes_as_source': [],
                        'partakes_as_target': [],
                        'entities': parts if len(parts) > 1 else [name]
                    })
                    # update is_character
                    flag = action.source_is_character if is_source else action.target_is_character
                    info['is_character'] = info['is_character'] or flag
                    # record participation
                    if is_source:
                        info['partakes_as_source'].append(action.action_id)
                    else:
                        info['partakes_as_target'].append(action.action_id)

        # build Node list
        nodes: List[Node] = []
        for name, info in sorted(node_info.items(), key=lambda x: x[0]):
            nodes.append(Node(
                node_name=name,
                is_single=info['is_single'],
                is_character=info['is_character'],
                partakes_as_source_in_actions=info['partakes_as_source'],
                partakes_as_target_in_actions=info['partakes_as_target'],
                entities_in_node_name=info['entities']
            ))

        return Graph(nodes=nodes, edges=edges)









