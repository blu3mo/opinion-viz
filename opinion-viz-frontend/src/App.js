import React, { useState } from 'react';
import ProjectView from './components/ProjectView';

function App() {
  const [projectId, setProjectId] = useState(1); // デフォルトで project_id=1 を表示

  const handleChange = (e) => {
    setProjectId(e.target.value);
  }

  return (
    <div style={{ margin: '2rem' }}>
      <h1>Opinion Visualization (FrontEnd)</h1>
      <p>Backend API: http://localhost:8000</p>
      <div>
        <label>Project ID: </label>
        <input type="number" value={projectId} onChange={handleChange} />
      </div>
      <ProjectView projectId={projectId} />
    </div>
  );
}

export default App;
