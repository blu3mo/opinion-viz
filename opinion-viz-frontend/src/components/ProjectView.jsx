// src/components/ProjectView.js
import React, { useEffect, useState } from 'react';

/**
 * 2D可視化 + クラスタ別表示サンプル
 */
function ProjectView({ projectId }) {
  // --- State ---
  // k: ユーザーが入力するクラスタ数
  const [k, setK] = useState(3);

  // clusters: サーバーから返却されるクラスタ配列
  const [clusters, setClusters] = useState([]);
  
  // Loading やエラー管理
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // hovered: ホバー時にツールチップ表示するための状態
  // { x, y, text } or null
  const [hovered, setHovered] = useState(null);

  // --- Fetch Clusters ---
  // k または projectId が変わったらクラスタ取得
  useEffect(() => {
    if (!projectId) return;

    setLoading(true);
    setError('');

    fetch(`http://localhost:8000/projects/${projectId}/clusters?k=${k}`)
      .then(res => {
        if (!res.ok) {
          throw new Error(`Failed to fetch clusters: status=${res.status}`);
        }
        return res.json();
      })
      .then(data => {
        // data.clusters の構造が { cluster_id, opinions } の形
        setClusters(data.clusters || []);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [projectId, k]);

  // --- クラスタ毎の意見を一つの配列にまとめ、2D描画のための情報に変換する ---
  // cluster_id がわかるように、意見に cluster_id を付与
  const mappedData = clusters.flatMap(cluster => {
    const { cluster_id, opinions } = cluster;
    return opinions.map(op => {
      const coords = op.coords_5d || [];
      const x = coords[0] || 0;  // x座標
      const y = coords[1] || 0;  // y座標

      return {
        clusterId: cluster_id,
        id: op.opinion_id,
        text: op.text,
        x,
        y,
      };
    });
  });

  // x,y の最小・最大
  const xs = mappedData.map(d => d.x);
  const ys = mappedData.map(d => d.y);
  const xMin = xs.length > 0 ? Math.min(...xs) : 0;
  const xMax = xs.length > 0 ? Math.max(...xs) : 1;
  const yMin = ys.length > 0 ? Math.min(...ys) : 0;
  const yMax = ys.length > 0 ? Math.max(...ys) : 1;

  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  // SVG領域
  const width = 600;
  const height = 600;
  const padding = 20;

  // スケール変換
  const scaleX = (val) => {
    return padding + ((val - xMin) / xRange) * (width - 2 * padding);
  };
  const scaleY = (val) => {
    return height - padding - ((val - yMin) / yRange) * (height - 2 * padding);
  };

  // クラスタID → 色 (適当なカラーマップを割り当てる例)
  // clusterId が 0,1,2,... のときに固定色を使うサンプル
  const clusterColors = [
    'rgb(255,0,0)',    // C0: Red
    'rgb(0,128,0)',    // C1: Green
    'rgb(0,0,255)',    // C2: Blue
    'rgb(255,165,0)',  // C3: Orange
    'rgb(255,0,255)',  // C4: Magenta
    'rgb(0,255,255)',  // C5: Aqua
    // さらに必要なら増やすか、ランダム生成などに変更
  ];

  const colorByClusterId = (clusterId) => {
    // clusterId が想定より大きい場合は mod などで折り返す
    return clusterColors[clusterId % clusterColors.length] || 'gray';
  };

  // ロード中 or エラー or データなし
  if (loading) return <p>Loading clusters...</p>;
  if (error) return <p style={{ color: 'red' }}>Error: {error}</p>;
  if (mappedData.length === 0) {
    return (
      <div>
        <p>No cluster data for project {projectId}.</p>
        <div style={{ marginTop: '1rem' }}>
          <label>Number of clusters (k): </label>
          <input
            type="number"
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            style={{ width: '60px' }}
            min={1}
          />
        </div>
      </div>
    );
  }

  return (
    <div style={{ marginTop: '1rem', position: 'relative', display: 'flex' }}>
      {/* 左側: 2Dプロット */}
      <div>
        <h2>Project ID: {projectId}</h2>
        <div style={{ marginBottom: '0.5rem' }}>
          <label>Number of clusters (k): </label>
          <input
            type="number"
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            style={{ width: '60px' }}
            min={1}
          />
        </div>

        <svg width={width} height={height} style={{ border: '1px solid #ccc' }}>
          {mappedData.map((d) => {
            const cx = scaleX(d.x);
            const cy = scaleY(d.y);
            const fillColor = colorByClusterId(d.clusterId);

            return (
              <circle
                key={d.id}
                cx={cx}
                cy={cy}
                r={6}
                fill={fillColor}
                stroke="#333"
                strokeWidth={1}
                onMouseEnter={() => {
                  setHovered({ x: cx, y: cy, text: d.text });
                }}
                onMouseLeave={() => setHovered(null)}
              />
            );
          })}
        </svg>

        {/* ホバー時のツールチップ */}
        {hovered && (
          <div
            style={{
              position: 'absolute',
              left: hovered.x + 10, // SVG 左上が (0,0)
              top: hovered.y + 10,
              background: 'white',
              border: '1px solid #555',
              borderRadius: 4,
              padding: '4px 8px',
              pointerEvents: 'none',
              whiteSpace: 'pre-wrap',
              maxWidth: '200px',
            }}
          >
            {hovered.text}
          </div>
        )}
      </div>

      {/* 右側: クラスタごとの意見一覧 */}
      <div style={{ marginLeft: '2rem' }}>
        <h3>Clustered Opinions</h3>
        {clusters.map((c) => (
          <div key={c.cluster_id} style={{ marginBottom: '1rem' }}>
            <h4 style={{ margin: '0.5rem 0' }}>
              Cluster {c.cluster_id}{' '}
              <span
                style={{
                  display: 'inline-block',
                  width: '12px',
                  height: '12px',
                  marginLeft: '8px',
                  backgroundColor: colorByClusterId(c.cluster_id),
                  border: '1px solid #333',
                }}
              />
            </h4>
            <ul style={{ marginLeft: '1.2rem' }}>
              {c.opinions.map((op) => (
                <li key={op.opinion_id}>{op.text}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ProjectView;
