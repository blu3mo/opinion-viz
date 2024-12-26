# main.py
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

from models.project import Project, Opinion
from services.embedding import get_embedding
from services.gpt import get_opinion_distance
from services.transform import fit_linear_transform, apply_linear_transform
from sklearn.cluster import KMeans

app = FastAPI()

# CORS設定を追加して、すべてのオリジン、メソッド、ヘッダーを許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 簡易的なインメモリDB
PROJECTS: Dict[int, Project] = {}
PROJECT_COUNTER = 1
OPINION_COUNTER = 1

# ----------------------------
# Pydantic Request Models
# ----------------------------
class CreateProjectRequest(BaseModel):
    name: str

class SetupProjectRequest(BaseModel):
    opinions: List[str]  # 初期意見(10件)

class AddOpinionRequest(BaseModel):
    text: str  # 新意見


@app.post("/projects")
def create_project(req: CreateProjectRequest):
    global PROJECT_COUNTER
    new_project = Project(
        id=PROJECT_COUNTER,
        name=req.name,
        opinions=[],
        weight=None,
        bias=None,
        embed_dim=None
    )
    PROJECTS[PROJECT_COUNTER] = new_project
    PROJECT_COUNTER += 1
    return {"project_id": new_project.id, "name": new_project.name}


@app.post("/projects/{project_id}/setup")
def setup_project(project_id: int, req: SetupProjectRequest):
    """
    10件の意見を登録:
      1) それぞれ埋め込みを作成 ("project_name: 意見")
      2) GPT-4でペアワイズ距離(0~1)を算出
      3) その距離をなるべく再現するように (d->5) の線形変換 (W, b) を学習
      4) (W, b) を保存し、10件に適用した5次元座標を保存
    """
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    project = PROJECTS[project_id]
    # 既存の意見をリセット
    project.opinions.clear()

    global OPINION_COUNTER

    # 1) 10件の意見を埋め込み化
    embeddings = []
    opinions_objs = []
    for text in req.opinions:
        op_id = OPINION_COUNTER
        OPINION_COUNTER += 1
        opinion_with_context = f"{project.name}: {text}"
        emb = get_embedding(project.name, opinion_with_context)  # => list of float
        opinions_objs.append(Opinion(
            id=op_id,
            text=text,
            embedding=emb,   # Pythonリスト
            coords_5d=None  # 後で埋める
        ))
        embeddings.append(emb)

    # numpy化
    embeddings_np = np.array(embeddings, dtype=np.float32)  # shape (10, d)
    d = embeddings_np.shape[1]
    project.embed_dim = d

    # 2) GPT-4 でペアワイズ距離(10x10)を取得
    n = len(opinions_objs)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            opinion_a_with_context = f"{project.name}: {opinions_objs[i].text}"
            opinion_b_with_context = f"{project.name}: {opinions_objs[j].text}"
            d_val = get_opinion_distance(
                opinion_a_with_context,
                opinion_b_with_context
            )
            dist_matrix[i, j] = d_val
            dist_matrix[j, i] = d_val

    # 3) (d->5) の線形変換 (W, b) を学習
    W, b = fit_linear_transform(embeddings_np, dist_matrix, n_components=5,
                                lr=1e-2, max_epochs=400, print_interval=100)
    # => shape of W: (d,5), b: (5,)

    # projectに保存
    project.weight = W.tolist()
    project.bias = b.tolist()

    # 4) 各意見の 5次元coordsを計算して保存
    for i, op in enumerate(opinions_objs):
        coord_5d = apply_linear_transform(np.array(op.embedding, dtype=np.float32), W, b)
        op.coords_5d = coord_5d.tolist()

    project.opinions = opinions_objs
    PROJECTS[project_id] = project

    return {
        "project_id": project_id,
        "message": "Project setup complete with 10 opinions and (W,b) learned."
    }


@app.post("/projects/{project_id}/opinions")
def add_opinion(project_id: int, req: AddOpinionRequest):
    """
    新意見を追加:
      1) "project_name: 新意見" を埋め込み化
      2) 学習済み(W, b)を適用して 5次元座標にする
      3) GPT-4に問い合わせる必要は無し！
    """
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    project = PROJECTS[project_id]
    if project.weight is None or project.bias is None:
        raise HTTPException(status_code=400, detail="Project has no trained transform (call setup first).")

    global OPINION_COUNTER
    new_op_id = OPINION_COUNTER
    OPINION_COUNTER += 1

    # 埋め込み
    opinion_with_context = f"{project.name}: {req.text}"
    emb = get_embedding(project.name, opinion_with_context)  # list of float
    W = np.array(project.weight, dtype=np.float32)  # (d,5)
    b = np.array(project.bias, dtype=np.float32)    # (5,)

    coords_5d = apply_linear_transform(np.array(emb, dtype=np.float32), W, b)
    coords_5d_list = coords_5d.tolist()

    new_opinion = Opinion(
        id=new_op_id,
        text=req.text,
        embedding=emb,
        coords_5d=coords_5d_list
    )

    project.opinions.append(new_opinion)
    PROJECTS[project_id] = project

    return {
        "project_id": project_id,
        "opinion_id": new_op_id,
        "coords_5d": coords_5d_list
    }


@app.get("/projects/{project_id}/opinions")
def get_opinions(project_id: int):
    """
    プロジェクトの意見一覧 (5次元座標を含む) を返す
    """
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    project = PROJECTS[project_id]

    return {
        "project_id": project_id,
        "opinions": [
            {
                "opinion_id": op.id,
                "text": op.text,
                "coords_5d": op.coords_5d
            }
            for op in project.opinions
        ]
    }

@app.get("/projects/{project_id}/clusters")
def cluster_opinions(project_id: int, k: int):
    """
    プロジェクト内の意見を、coords_5d に基づき k-means でクラスタリングし、
    クラスタIDごとに意見リストを返す。
    クエリパラメータ ?k=3 のように指定すると、k=3クラスタに分類する。
    """
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    project = PROJECTS[project_id]
    if not project.opinions:
        raise HTTPException(status_code=400, detail="No opinions found in this project.")
    
    # coords_5d が無い場合は setup 未完了かもしれない
    # あるいは coords_5d を使用しないプロジェクト？
    # とりあえず例では coords_5d が全員ある前提で進める
    coords_list = []
    for op in project.opinions:
        if not op.coords_5d:
            raise HTTPException(status_code=400, detail=f"Opinion {op.id} has no coords_5d.")
        coords_list.append(op.coords_5d)
    
    coords_array = np.array(coords_list, dtype=np.float32)  # shape: (N, 5)

    # k-means
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(coords_array)  # shape: (N,)

    # cluster_id -> opinions
    clusters = {}
    for i, op in enumerate(project.opinions):
        label = int(labels[i])
        if label not in clusters:
            clusters[label] = []
        clusters[label].append({
            "opinion_id": op.id,
            "text": op.text,
            "coords_5d": op.coords_5d
        })
    
    # 出力形式: cluster_id がキーだと JSON としては扱いづらい場合もあるので、
    # "clusters": [ { "cluster_id":..., "opinions": [...] }, ... ] の構造にする例
    cluster_list = []
    for cluster_id, op_list in clusters.items():
        cluster_list.append({
            "cluster_id": cluster_id,
            "opinions": op_list
        })

    return {
        "project_id": project_id,
        "k": k,
        "clusters": cluster_list
    }