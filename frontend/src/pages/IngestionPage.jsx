import { useState, useEffect } from 'react'
import {
  FileText, Scissors, Sparkles, Download,
  Database, CheckCircle, Play, RotateCcw, Upload
} from 'lucide-react'

const ICON_MAP = {
  'file-text':    <FileText size={20} />,
  'scissors':     <Scissors size={20} />,
  'sparkles':     <Sparkles size={20} />,
  'download':     <Download size={20} />,
  'database':     <Database size={20} />,
  'check-circle': <CheckCircle size={20} />
}

// Static fallback in case API is not running
const STATIC_INGESTION_STAGES = [
  { id: 'loader',       name: 'Document Loader',       description: 'Scans Data/ directory for PDFs & TXT files, extracts raw text using PyMuPDF.', model: null, icon: 'file-text' },
  { id: 'chunking',     name: 'Semantic Chunking',      description: 'Splits documents at semantic breakpoints using cosine similarity of sentence embeddings.', model: 'nvidia/nv-embedqa-e5-v5', icon: 'scissors' },
  { id: 'enrichment',   name: 'Metadata Enrichment',    description: 'Extracts chunk title, entities, QA pairs and compliance mandates via async LLM batch calls (≤40 RPM).', model: 'meta/llama-3.1-8b-instruct', icon: 'sparkles' },
  { id: 'export',       name: 'JSON/JSONL Export',      description: 'Exports all enriched chunks to processed_documents.json and .jsonl artifacts.', model: null, icon: 'download' },
  { id: 'indexing',     name: 'Qdrant Vector Indexing', description: 'Bulk-inserts enriched chunks as dense vectors into a persistent Qdrant collection.', model: 'nvidia/llama-nemotron-embed-1b-v2', icon: 'database' },
  { id: 'verification', name: 'Verification Search',    description: 'Runs test queries against the freshly indexed collection to confirm retrieval quality.', model: null, icon: 'check-circle' }
]

const STATIC_DOC_STAGES = [
  { id: 'scan',    name: 'File Scanner',             description: 'Recursively scans the Data/ directory and filters for supported file types (PDF, TXT).', model: null, icon: 'file-text' },
  { id: 'load',    name: 'Document Loading',         description: 'Loads raw text content from each file using PyMuPDF for PDFs and native loaders for TXT.', model: null, icon: 'download' },
  { id: 'chunk',   name: 'Semantic Chunking',        description: 'Uses NVIDIA embedding model to compute sentence-level cosine similarities and detect semantic breakpoints.', model: 'nvidia/nv-embedqa-e5-v5', icon: 'scissors' },
  { id: 'enrich',  name: 'Chunk-Level Enrichment',   description: 'Extracts title, domain, summary, entities, synthetic QA pairs, and compliance mandates per chunk using LLM.', model: 'meta/llama-3.1-8b-instruct', icon: 'sparkles' },
  { id: 'ratelim', name: 'Rate Limiter (≤40 RPM)',   description: 'Async sliding-window rate limiter batches API calls in groups of 5 to stay under NVIDIA API limits.', model: null, icon: 'check-circle' },
  { id: 'artifact','name': 'Artifact Export',        description: 'Saves all enriched chunks as processed_documents.json and .jsonl for audit trail and indexing.', model: null, icon: 'download' }
]

export default function IngestionPage() {
  const [tab, setTab]             = useState('ingestion')
  const [stages, setStages]       = useState([])
  const [activeIdx, setActiveIdx] = useState(-1)
  const [running, setRunning]     = useState(false)
  const [doneIdxs, setDoneIdxs]   = useState(new Set())
  const [files, setFiles]         = useState(null)
  const [uploading, setUploading] = useState(false)
  const [ingestStatus, setIngestStatus] = useState('')

  useEffect(() => {
    setStages(tab === 'ingestion' ? STATIC_INGESTION_STAGES : STATIC_DOC_STAGES)
    setActiveIdx(-1)
    setDoneIdxs(new Set())
    setRunning(false)
  }, [tab])

  const runSimulation = () => {
    if (running) return
    setRunning(true)
    setActiveIdx(0)
    setDoneIdxs(new Set())

    let idx = 0
    const tick = () => {
      setActiveIdx(idx)
      setTimeout(() => {
        setDoneIdxs(prev => new Set([...prev, idx]))
        idx++
        if (idx < stages.length) {
          setTimeout(tick, 300)
        } else {
          setActiveIdx(-1)
          setRunning(false)
        }
      }, 900)
    }
    tick()
  }

  const reset = () => {
    setActiveIdx(-1)
    setDoneIdxs(new Set())
    setRunning(false)
    setIngestStatus('')
  }

  const handleFileChange = (e) => {
    setFiles(e.target.files)
  }

  const handleUploadAndIngest = async () => {
    if (!files || files.length === 0) return
    setUploading(true)
    setIngestStatus('Uploading files...')
    
    const formData = new FormData()
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i])
    }
    
    try {
      const uploadRes = await fetch('http://localhost:8080/api/upload', {
        method: 'POST',
        body: formData
      })
      if (!uploadRes.ok) throw new Error('Upload failed')
      
      setIngestStatus('Starting pipeline...')
      const ingestRes = await fetch('http://localhost:8080/api/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ skip_indexing: false })
      })
      if (!ingestRes.ok) throw new Error('Ingestion failed to start')
      
      setIngestStatus('Pipeline running in background...')
      setFiles(null)
      // Clear file input
      const fileInput = document.getElementById('file-upload')
      if (fileInput) fileInput.value = ''
      
      runSimulation()
    } catch (err) {
      console.error(err)
      setIngestStatus(`Error: ${err.message}`)
    } finally {
      setUploading(false)
    }
  }

  const progress = stages.length > 0
    ? Math.round((doneIdxs.size / stages.length) * 100) : 0

  return (
    <>
      {/* Header */}
      <div className="page-header">
        <div>
          <div className="page-title">Pipeline Visualizer</div>
          <div className="page-subtitle">
            Animated visualization of the ingestion & document processing stages
          </div>
        </div>
        <span className="badge badge-cyan">Read-Only View</span>
      </div>

      <div className="ingestion-page">
        {/* Stats */}
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-value">6</div>
            <div className="stat-label">Pipeline Stages</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">40</div>
            <div className="stat-label">Max RPM (Rate Limit)</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">Qdrant</div>
            <div className="stat-label">Vector Store</div>
          </div>
        </div>

        {/* Tabs */}
        <div className="pipeline-tabs">
          <button
            className={`pipeline-tab ${tab === 'ingestion' ? 'active' : ''}`}
            onClick={() => setTab('ingestion')}
          >
            Ingestion Pipeline
          </button>
          <button
            className={`pipeline-tab ${tab === 'document' ? 'active' : ''}`}
            onClick={() => setTab('document')}
          >
            Document Pipeline
          </button>
        </div>

        {/* Controls */}
        <div className="run-btn-wrap" style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <input 
            type="file" 
            id="file-upload" 
            multiple 
            accept=".pdf,.txt,.md" 
            onChange={handleFileChange}
            disabled={running || uploading}
            style={{ 
              background: 'var(--bg-card)', 
              color: 'var(--text-main)', 
              border: '1px solid var(--border)', 
              padding: '8px', 
              borderRadius: '6px' 
            }}
          />
          <button 
            className="btn btn-cyan" 
            onClick={handleUploadAndIngest} 
            disabled={!files || files.length === 0 || running || uploading}
          >
            <Upload size={14} />
            {uploading ? 'Uploading...' : 'Upload & Ingest'}
          </button>
          
          <button className="btn btn-cyan" onClick={runSimulation} disabled={running}>
            <Play size={14} />
            {running ? 'Running…' : 'Simulate Run'}
          </button>
          <button className="btn btn-cyan" onClick={reset} disabled={running}>
            <RotateCcw size={14} />
            Reset
          </button>
        </div>
        {ingestStatus && (
          <div style={{ marginTop: '1rem', color: 'var(--cyan)', fontSize: '0.9rem' }}>
            {ingestStatus}
          </div>
        )}

        {/* Progress */}
        <div className="pipeline-progress">
          <div className="progress-label">
            <span>Pipeline Progress</span>
            <span>{doneIdxs.size} / {stages.length} stages</span>
          </div>
          <div className="progress-bar-track">
            <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
          </div>
        </div>

        {/* Stage flow */}
        <div className="stage-flow">
          {stages.map((stage, i) => {
            const isDone   = doneIdxs.has(i)
            const isActive = activeIdx === i
            return (
              <div key={stage.id}>
                <div
                  className={`stage-card ${isActive ? 'active' : ''} ${isDone ? 'completed' : ''}`}
                  style={{ animationDelay: `${i * 0.05}s` }}
                >
                  <div className="stage-icon-wrap">
                    {ICON_MAP[stage.icon]}
                  </div>
                  <div className="stage-meta">
                    <div className="stage-name">{stage.name}</div>
                    <div className="stage-desc">{stage.description}</div>
                    {stage.model && (
                      <div className="stage-model">▸ {stage.model}</div>
                    )}
                  </div>
                  <div className="stage-status">
                    {isDone && <span className="badge badge-green">Done</span>}
                    {isActive && <span className="badge badge-cyan">Running</span>}
                    {!isDone && !isActive && <span className="badge" style={{color:'var(--text-muted)',background:'transparent',border:'1px solid var(--border)'}}>Idle</span>}
                  </div>
                </div>
                {i < stages.length - 1 && (
                  <div className="stage-connector" />
                )}
              </div>
            )
          })}
        </div>
      </div>
    </>
  )
}
