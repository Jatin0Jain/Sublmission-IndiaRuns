import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Upload, Search, Users, Activity, Target, Zap, LayoutDashboard, ChevronRight, AlertTriangle, CheckCircle, Database } from 'lucide-react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';

import './index.css';

const API_URL = '/api';

function App() {
  const [jdText, setJdText] = useState("We are looking for a Senior Backend Engineer with deep expertise in Python, FastAPI, and Postgres. You should have experience with Docker and AWS. 5+ years of experience required.");
  const [skillWeight, setSkillWeight] = useState(0.6);
  const [growthWeight, setGrowthWeight] = useState(0.3);
  const [activityWeight, setActivityWeight] = useState(0.2);
  const [demoMode, setDemoMode] = useState(true);
  
  const [candidates, setCandidates] = useState([]);
  const [times, setTimes] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploadMode, setUploadMode] = useState("merge");

  const handleSearch = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${API_URL}/search`, {
        jd_text: jdText,
        skill_weight: skillWeight,
        growth_weight: growthWeight,
        activity_weight: activityWeight,
        demo_mode: demoMode
      });
      setCandidates(res.data.candidates);
      setTimes(res.data.times);
    } catch (err) {
      alert(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setUploading(true);
    setUploadStatus("Uploading & generating embeddings (This may take a few minutes)...");
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('mode', uploadMode);
    
    try {
      const res = await axios.post(`${API_URL}/upload_dataset`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setUploadStatus(res.data.message);
      setDemoMode(false); // Automatically switch off demo mode
    } catch (err) {
      setUploadStatus(`Error: ${err.response?.data?.detail || err.message}`);
    } finally {
      setUploading(false);
    }
  };

  const handleResetDataset = async () => {
    setUploading(true);
    setUploadStatus("Restoring original 100k dataset...");
    try {
      const res = await axios.post(`${API_URL}/reset_dataset`);
      setUploadStatus(res.data.message);
    } catch (err) {
      setUploadStatus(`Error: ${err.response?.data?.detail || err.message}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="app-container">
      {/* SIDEBAR */}
      <div className="sidebar">
        <div>
          <h1 className="nova-title">NovaSearch</h1>
          <p className="nova-sub">Enterprise Talent Intelligence</p>
        </div>

        <div>
          <div className="sb-label">📋 Job Description</div>
          <textarea 
            value={jdText} 
            onChange={e => setJdText(e.target.value)} 
          />
        </div>

        <div className="slider-group">
          <div className="sb-label">⚙️ Algorithm Tuning</div>
          <div className="slider-item">
            <div className="slider-header">
              <span>Skill Match Importance</span>
              <span>{skillWeight.toFixed(2)}</span>
            </div>
            <input type="range" min="0" max="1" step="0.05" value={skillWeight} onChange={e => setSkillWeight(parseFloat(e.target.value))} />
          </div>
          <div className="slider-item">
            <div className="slider-header">
              <span>Career Growth Velocity</span>
              <span>{growthWeight.toFixed(2)}</span>
            </div>
            <input type="range" min="0" max="1" step="0.05" value={growthWeight} onChange={e => setGrowthWeight(parseFloat(e.target.value))} />
          </div>
          <div className="slider-item">
            <div className="slider-header">
              <span>Recent Activity</span>
              <span>{activityWeight.toFixed(2)}</span>
            </div>
            <input type="range" min="0" max="1" step="0.05" value={activityWeight} onChange={e => setActivityWeight(parseFloat(e.target.value))} />
          </div>
        </div>
        
        <div>
          <div className="sb-label">📂 Custom Dataset</div>
          <div style={{ background: 'var(--bg-input)', padding: '1rem', borderRadius: '8px', border: '1px dashed var(--border)', textAlign: 'center' }}>
            <Database size={24} color="var(--text-2)" style={{marginBottom: '8px'}} />
            <p style={{fontSize: '.8rem', color: 'var(--text-1)', margin: '0 0 10px'}}>Upload a .CSV or .JSON file</p>
            
            <div style={{display: 'flex', justifyContent: 'center', gap: '10px', marginBottom: '10px', fontSize: '.75rem'}}>
              <label style={{cursor: 'pointer'}}>
                <input type="radio" name="uploadMode" value="merge" checked={uploadMode === 'merge'} onChange={() => setUploadMode('merge')} /> Merge
              </label>
              <label style={{cursor: 'pointer'}}>
                <input type="radio" name="uploadMode" value="replace" checked={uploadMode === 'replace'} onChange={() => setUploadMode('replace')} /> Replace
              </label>
            </div>

            <input type="file" id="dataset-upload" style={{display: 'none'}} accept=".csv,.json,.jsonl,.parquet" onChange={handleFileUpload} />
            <label htmlFor="dataset-upload" className="btn-primary" style={{fontSize: '.85rem', padding: '8px', background: 'var(--bg-surface)'}}>
              {uploading ? 'Processing...' : 'Upload File'}
            </label>
            <button 
              className="btn-primary" 
              onClick={handleResetDataset} 
              disabled={uploading}
              style={{fontSize: '.75rem', padding: '6px', background: 'transparent', border: '1px solid var(--border)', marginTop: '8px', color: 'var(--text-1)'}}
            >
              ↻ Restore Default Database
            </button>
            {uploadStatus && <p style={{fontSize: '.75rem', marginTop: '10px', color: uploadStatus.includes('Error') ? 'var(--red)' : 'var(--green)'}}>{uploadStatus}</p>}
          </div>
        </div>

        <div style={{ marginTop: 'auto' }}>
          <label className="toggle-group" style={{marginBottom: '1rem', cursor: 'pointer'}}>
            <input type="checkbox" checked={demoMode} onChange={e => setDemoMode(e.target.checked)} />
            🚀 Offline Demo Mode
          </label>
          <button className="btn-primary" onClick={handleSearch} disabled={loading || uploading}>
            {loading ? 'Analyzing...' : <><Search size={18} /> Run AI Search</>}
          </button>
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div className="main-content">
        {candidates.length > 0 ? (
          <div>
            {/* KPIs */}
            <div className="kpi-strip">
              <div className="kpi-card">
                <div className="kpi-label">Candidates</div>
                <div className="kpi-value">Pool</div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">Top Matches</div>
                <div className="kpi-value accent">{candidates.length}</div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">Avg. Score</div>
                <div className="kpi-value green">
                  {Math.round(candidates.reduce((a,c) => a + c.score, 0) / candidates.length)}
                </div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">Total Latency</div>
                <div className="kpi-value">{(times.vector + times.ai).toFixed(2)}s</div>
              </div>
            </div>
            
            {/* Candidate List */}
            {candidates.map((c, i) => (
              <div className="c-card" key={c.candidate_id || i} style={{animationDelay: `${i * 0.1}s`}}>
                <div className="c-header">
                  <div className="c-avatar">{c.name.charAt(0)}</div>
                  <div className="c-meta">
                    <div className="c-name"><span className="c-rank">#{c.rank}</span> {c.name}</div>
                    <div className="c-role">{c.title}</div>
                  </div>
                  <div className="c-score-box">
                    <div className="c-score" style={{color: c.score >= 80 ? 'var(--green)' : c.score >= 60 ? 'var(--amber)' : 'var(--red)'}}>
                      {c.score}
                    </div>
                    <div className="c-score-label">/ 100</div>
                  </div>
                </div>
                
                <div className="score-bar-track">
                  <div className="score-bar-fill" style={{
                    width: `${c.score}%`, 
                    background: `linear-gradient(90deg, transparent, ${c.score >= 80 ? 'var(--green)' : c.score >= 60 ? 'var(--amber)' : 'var(--red)'})`
                  }}></div>
                </div>
                
                <div className="skills-row">
                  {c.skills_matched.map(s => <span key={s} className="skill hit">✓ {s}</span>)}
                  {c.skills_missing.map(s => <span key={s} className="skill miss">✗ {s}</span>)}
                </div>
                
                <div className="narr">
                  <div className="narr-label"><Zap size={14} /> AI Analysis</div>
                  {c.narrative}
                </div>
                
                <div className="flags">
                  <div className="flag warn"><AlertTriangle size={16} /> {c.caution}</div>
                  <div className="flag good"><CheckCircle size={16} /> {c.standout}</div>
                </div>

                <div className="card-footer">
                  <div className="radar-container">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart cx="50%" cy="50%" outerRadius="70%" data={[
                        { subject: 'Skill', A: c.radar.skill },
                        { subject: 'Exp', A: c.radar.experience },
                        { subject: 'Growth', A: c.radar.growth },
                        { subject: 'Culture', A: c.radar.culture },
                        { subject: 'Avail', A: c.radar.availability },
                      ]}>
                        <PolarGrid stroke="rgba(255,255,255,0.05)" />
                        <PolarAngleAxis dataKey="subject" tick={{fill: '#9898b8', fontSize: 11}} />
                        <Radar name="Candidate" dataKey="A" stroke="#7c6aff" fill="#7c6aff" fillOpacity={0.2} />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="interview-prep">
                    <h4><Target size={16} /> Interview Prep Questions</h4>
                    {c.interview_questions && c.interview_questions.length > 0 ? (
                      <ol>
                        {c.interview_questions.map((q, idx) => <li key={idx}>{q}</li>)}
                      </ol>
                    ) : (
                      <p style={{fontSize: '.85rem', color: 'var(--text-2)'}}>Run without demo mode to generate targeted questions based on candidate weak spots.</p>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="hero">
            <div className="hero-icon">✦</div>
            <h2>Awaiting Intelligence</h2>
            <p>Paste your <b>Job Description</b> in the sidebar, optionally upload your own dataset, tune the algorithm weights, and click <b>Run AI Search</b>.</p>
            <div className="stats-row">
              <div className="stat-block">
                <div className="stat-number">100k</div>
                <div className="stat-label">Profiles indexed</div>
              </div>
              <div className="stat-block">
                <div className="stat-number">384-D</div>
                <div className="stat-label">Vector space</div>
              </div>
              <div className="stat-block">
                <div className="stat-number">&lt;2s</div>
                <div className="stat-label">End-to-end</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
