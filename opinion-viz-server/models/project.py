# models/project.py
from typing import List, Optional
from pydantic import BaseModel

class Opinion(BaseModel):
    id: int
    text: str
    embedding: Optional[list] = None   # 元の埋め込みベクトル
    coords_5d: Optional[list] = None  # 学習済み (W, b) を適用した 5次元座標

class Project(BaseModel):
    id: int
    name: str
    opinions: List[Opinion]  # このプロジェクトが保持する意見
    weight: Optional[list] = None     # (d x 5) の W
    bias: Optional[list] = None       # (5,) の b
    embed_dim: Optional[int] = None   # 埋め込み次元 (d)
