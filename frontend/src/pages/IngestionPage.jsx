import { useState, useEffect, useRef } from 'react'
import {
  FileText, Scissors, Sparkles, Download,
  Database, CheckCircle, Play, RotateCcw, Upload,
  File, X, AlertCircle, RefreshCw, Layers
} from 'lucide-react'

const ICON_MAP = {
  'file-text':    <FileText size={20} />,
  'scissors':     <Scissors size={20} />,
  'sparkles':     <Sparkles size={20} />,
  'download':     <Download size={20} />,
  'database':     <Database size={20} />,
  'check-circle': <CheckCircle size={20} />
}

const UNIFIED_STAGES = [
  {
    id: 'loader',
    name: '1. Document Loader & Parser',
    description: 'Scans uploaded files & Data/ folder, extracts raw text via PyMuPDF for PDFs and native loaders for TXT/MD.',
    model: 'PyMuPDF Parser',
    icon: 'file-text'
  },
  {
    id: 'chunking',
    name: '2. Semantic Chunking',
    description: 'Computes sentence embeddings to identify semantic breakpoints, splitting documents contextually rather than arbitrarily.',
    model: 'nvidia/nv-embedqa-e5-v5',
    icon: 'scissors'
  },
  {
    id: 'enrichment',
    name: '3. LLM Metadata Enrichment',
    description: 'Extracts chunk title, entities, QA pairs, topics and compliance mandates via async LLM batch calls (≤40 RPM).',
    model: 'meta/llama-3.1-8b-instruct',
    icon: 'sparkles'
  },
  {
    id: 'export',
    name: '4. Artifact Export & Append',
    description: 'Saves and appends all enriched chunks to processed_documents.json and .jsonl artifacts with continuous chunk IDs.',
    model: 'JSON & JSONL',
    icon: 'download'
  },
  {
    id: 'indexing',
    name: '5. Qdrant Vector Indexing',
    description: 'Bulk-inserts enriched chunks as dense vectors into the persistent local Qdrant collection.',
    model: 'nvidia/llama-nemotron-embed-1b-v2',
    icon: 'database'
  },
  {
    id: 'verification',
    name: '6. Verification & Live Reload',
    description: 'Runs automated verification search queries and instantly reloads the in-memory RAG pipeline for real-time querying.',
    model: 'Hybrid Verification',
    icon: 'check-circle'
  }
]

export default function IngestionPage() {
  const [selectedFiles, setSelectedFiles] = useState([])
  const [isDragging, setIsDragging]       = useState(false)
  const [activeIdx, setActiveIdx]         = useState(-1)
  const [running, setRunning]             = useState(false)
  const [doneIdxs, setDoneIdxs]           = useState(new Set())
  const [statusMessage, setStatusMessage] = useState('Select document files (.pdf, .txt, .md) to process.')
  const [isLiveRunning, setIsLiveRunning] = useState(false)
  const [docStats, setDocStats]           = useState({ total_chunks: 0, sources: [] })
  const [lastResult, setLastResult]       = useState(null)
  const fileInputRef                      = useRef(null)

  // Fetch document inventory
  const fetchDocStats = () => {
    fetch('http://localhost:8080/api/documents')
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data) setDocStats(data)
      })
      .catch(() => {})
  }

  useEffect(() => {
    fetchDocStats()
  }, [])

  // Poll live backend ingestion status
  useEffect(() => {
    if (!isLiveRunning) return

    const interval = setInterval(async () => {
      try {
        const res = await fetch('http://localhost:8080/api/ingest/status')
        if (!res.ok) return
        const data = await res.json()

        if (data.status === 'running') {
          setStatusMessage(data.message || 'Pipeline processing...')
          if (data.stage_index !== undefined) {
            setActiveIdx(data.stage_index)
            const completed = new Set()
            for (let i = 0; i < data.stage_index; i++) completed.add(i)
            setDoneIdxs(completed)
          }
        } else if (data.status === 'completed') {
          setIsLiveRunning(false)
          setRunning(false)
          setActiveIdx(-1)
          const allDone = new Set(UNIFIED_STAGES.map((_, i) => i))
          setDoneIdxs(allDone)
          setStatusMessage(`✅ Ingestion Complete: ${data.message}`)
          setLastResult(data)
          fetchDocStats()
        } else if (data.status === 'error') {
          setIsLiveRunning(false)
          setRunning(false)
          setStatusMessage(`❌ Error: ${data.error || 'Ingestion failed'}`)
        }
      } catch (err) {
        console.error('Status poll error:', err)
      }
    }, 1200)

    return () => clearInterval(interval)
  }, [isLiveRunning])

  // File selection handlers
  const handleFileSelect = (files) => {
    if (!files || files.length === 0) return
    const newFiles = Array.from(files).filter(f => 
      f.name.endsWith('.pdf') || f.name.endsWith('.txt') || f.name.endsWith('.md')
    )
    setSelectedFiles(prev => [...prev, ...newFiles])
    setStatusMessage(`${newFiles.length} file(s) selected. Ready to upload & process.`)
  }

  const removeFile = (index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index))
  }

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / 1048576).toFixed(1) + ' MB'
  }

  // Upload and execute unified pipeline
  const handleUploadAndRun = async () => {
    if (selectedFiles.length === 0 || running) return

    setRunning(true)
    setIsLiveRunning(true)
    setActiveIdx(0)
    setDoneIdxs(new Set())
    setLastResult(null)
    setStatusMessage('Uploading files to staging...')

    const formData = new FormData()
    selectedFiles.forEach(file => {
      formData.append('files', file)
    })

    try {
      const uploadRes = await fetch('http://localhost:8080/api/upload', {
        method: 'POST',
        body: formData
      })
      if (!uploadRes.ok) {
        const err = await uploadRes.json().catch(() => ({}))
        throw new Error(err.error || 'Upload failed.')
      }

      setStatusMessage('Triggering combined ingestion pipeline...')
      const ingestRes = await fetch('http://localhost:8080/api/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ skip_indexing: false })
      })

      if (!ingestRes.ok) {
        const err = await ingestRes.json().catch(() => ({}))
        throw new Error(err.message || 'Ingestion trigger failed.')
      }

      setSelectedFiles([])
      if (fileInputRef.current) fileInputRef.current.value = ''
      setStatusMessage('Unified pipeline running across all 6 stages...')

    } catch (err) {
      console.error(err)
      setIsLiveRunning(false)
      setRunning(false)
      setStatusMessage(`Error: ${err.message}`)
    }
  }

  // Visual simulation
  const runSimulation = () => {
    if (running) return
    setRunning(true)
    setIsLiveRunning(false)
    setActiveIdx(0)
    setDoneIdxs(new Set())
    setLastResult(null)
    setStatusMessage('Simulating end-to-end ingestion pipeline...')

    let idx = 0
    const tick = () => {
      setActiveIdx(idx)
      setTimeout(() => {
        setDoneIdxs(prev => new Set([...prev, idx]))
        idx++
        if (idx < UNIFIED_STAGES.length) {
          setTimeout(tick, 350)
        } else {
          setActiveIdx(-1)
          setRunning(false)
          setStatusMessage('✅ Simulation finished successfully across all 6 stages.')
        }
      }, 1000)
    }
    tick()
  }

  const resetPipeline = () => {
    setActiveIdx(-1)
    setDoneIdxs(new Set())
    setRunning(false)
    setIsLiveRunning(false)
    setSelectedFiles([])
    setLastResult(null)
    setStatusMessage('Pipeline ready for document ingestion.')
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const progress = Math.round((doneIdxs.size / UNIFIED_STAGES.length) * 100)

  return (
    <>
      {/* Header */}
      <div className="page-header">
        <div>
          <div className="page-title">Unified Ingestion & Document Pipeline</div>
          <div className="page-subtitle">
            Upload documents to trigger automated chunking, LLM enrichment, Qdrant vector indexing & live reload
          </div>
        </div>
        <button 
          onClick={fetchDocStats}
          className="btn btn-ghost"
          style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px' }}
        >
          <RefreshCw size={14} /> Refresh Inventory
        </button>
      </div>

      <div className="ingestion-page" style={{ padding: '24px 28px', maxWidth: '1200px', margin: '0 auto' }}>
        {/* Stats Grid */}
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-value">{UNIFIED_STAGES.length}</div>
            <div className="stat-label">Connected Stages</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{docStats.total_chunks || 0}</div>
            <div className="stat-label">Total Chunks in DB</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{docStats.sources?.length || 0}</div>
            <div className="stat-label">Indexed Documents</div>
          </div>
        </div>

        {/* Upload & Controls Section */}
        <div style={{
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-md)',
          padding: '20px',
          marginBottom: '24px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <Layers size={18} color="var(--cyan)" />
            <h3 style={{ fontFamily: 'var(--font-heading)', fontSize: '14px', textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--text-primary)', margin: 0 }}>
              Document Uploader & Combined Pipeline Trigger
            </h3>
          </div>

          {/* Drag and Drop Zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setIsDragging(false)
              handleFileSelect(e.dataTransfer.files)
            }}
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: `2px dashed ${isDragging ? 'var(--cyan)' : 'var(--border)'}`,
              borderRadius: 'var(--radius-sm)',
              padding: '28px 20px',
              textAlign: 'center',
              cursor: 'pointer',
              background: isDragging ? 'var(--cyan-dim)' : 'var(--bg-secondary)',
              transition: 'all 0.2s',
              marginBottom: '16px'
            }}
          >
            <input
              type="file"
              ref={fileInputRef}
              multiple
              accept=".pdf,.txt,.md"
              onChange={(e) => handleFileSelect(e.target.files)}
              style={{ display: 'none' }}
            />
            <Upload size={32} color="var(--cyan)" style={{ margin: '0 auto 10px', opacity: 0.8 }} />
            <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px' }}>
              Click to browse or drag & drop documents here
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Supports PDF, TXT, and Markdown files (.pdf, .txt, .md)
            </div>
          </div>

          {/* Selected Files List */}
          {selectedFiles.length > 0 && (
            <div style={{ marginBottom: '16px' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--cyan)', marginBottom: '8px' }}>
                Selected Files ({selectedFiles.length}):
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {selectedFiles.map((file, idx) => (
                  <div
                    key={idx}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      padding: '6px 12px',
                      background: 'var(--bg-secondary)',
                      border: '1px solid var(--cyan-glow)',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: '12px',
                      color: 'var(--text-primary)'
                    }}
                  >
                    <File size={14} color="var(--cyan)" />
                    <span>{file.name}</span>
                    <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>({formatFileSize(file.size)})</span>
                    <button
                      onClick={(e) => { e.stopPropagation(); removeFile(idx) }}
                      style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: 0 }}
                    >
                      <X size={14} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
            <button
              className="btn btn-cyan"
              onClick={handleUploadAndRun}
              disabled={selectedFiles.length === 0 || running}
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <Upload size={14} />
              {running && isLiveRunning ? 'Processing Live...' : 'Upload & Run Combined Pipeline'}
            </button>

            <button
              className="btn btn-ghost"
              onClick={runSimulation}
              disabled={running}
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <Play size={14} />
              {running && !isLiveRunning ? 'Simulating…' : 'Simulate Animation'}
            </button>

            <button
              className="btn btn-ghost"
              onClick={resetPipeline}
              disabled={running}
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <RotateCcw size={14} /> Reset
            </button>
          </div>

          {/* Status Message */}
          <div style={{
            marginTop: '14px',
            padding: '10px 14px',
            background: 'var(--bg-secondary)',
            borderLeft: '3px solid var(--cyan)',
            borderRadius: '0 var(--radius-sm) var(--radius-sm) 0',
            fontSize: '12px',
            color: 'var(--text-secondary)'
          }}>
            {statusMessage}
          </div>
        </div>

        {/* Live Result Banner */}
        {lastResult && (
          <div style={{
            padding: '16px',
            background: 'rgba(0,255,157,0.08)',
            border: '1px solid var(--green)',
            borderRadius: 'var(--radius-md)',
            marginBottom: '24px',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '12px'
          }}>
            <CheckCircle size={22} color="var(--green)" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--green)', marginBottom: '4px' }}>
                Pipeline Successfully Completed!
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-primary)', lineHeight: 1.5 }}>
                • <strong>{lastResult.chunks_count || 0} chunks</strong> extracted, enriched via Llama-3.1, exported and indexed into Qdrant.<br />
                • Active retrieval pipeline automatically refreshed — new documents are immediately queryable in Chat!
              </div>
            </div>
          </div>
        )}

        {/* Progress Bar */}
        <div className="pipeline-progress" style={{ marginBottom: '24px' }}>
          <div className="progress-label" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)' }}>End-to-End Progress</span>
            <span style={{ fontSize: '12px', color: 'var(--cyan)', fontFamily: 'var(--font-heading)' }}>
              {doneIdxs.size} / {UNIFIED_STAGES.length} Stages ({progress}%)
            </span>
          </div>
          <div className="progress-bar-track" style={{ height: '6px', background: 'rgba(255,255,255,0.08)', borderRadius: '3px', overflow: 'hidden' }}>
            <div
              className="progress-bar-fill"
              style={{
                width: `${progress}%`,
                height: '100%',
                background: 'linear-gradient(90deg, var(--cyan), var(--purple))',
                boxShadow: '0 0 10px var(--cyan-glow)',
                transition: 'width 0.4s ease'
              }}
            />
          </div>
        </div>

        {/* Connected Stage Flow */}
        <div className="stage-flow">
          {UNIFIED_STAGES.map((stage, i) => {
            const isDone   = doneIdxs.has(i)
            const isActive = activeIdx === i
            return (
              <div key={stage.id}>
                <div
                  className={`stage-card ${isActive ? 'active' : ''} ${isDone ? 'completed' : ''}`}
                  style={{
                    animationDelay: `${i * 0.05}s`,
                    borderColor: isActive ? 'var(--cyan)' : isDone ? 'rgba(0,255,157,0.4)' : 'var(--border)',
                    boxShadow: isActive ? '0 0 20px rgba(10,240,255,0.2)' : 'none'
                  }}
                >
                  <div className="stage-icon-wrap" style={{
                    color: isActive ? 'var(--cyan)' : isDone ? 'var(--green)' : 'var(--text-muted)'
                  }}>
                    {ICON_MAP[stage.icon]}
                  </div>
                  <div className="stage-meta">
                    <div className="stage-name" style={{ color: isActive ? 'var(--cyan)' : 'var(--text-primary)' }}>
                      {stage.name}
                    </div>
                    <div className="stage-desc">{stage.description}</div>
                    {stage.model && (
                      <div className="stage-model" style={{ marginTop: '4px', fontSize: '11px', color: 'var(--purple)' }}>
                        ▸ {stage.model}
                      </div>
                    )}
                  </div>
                  <div className="stage-status">
                    {isDone && <span className="badge badge-green">Done</span>}
                    {isActive && <span className="badge badge-cyan" style={{ animation: 'pulse 1s infinite' }}>Running</span>}
                    {!isDone && !isActive && (
                      <span className="badge" style={{ color: 'var(--text-muted)', background: 'transparent', border: '1px solid var(--border)' }}>
                        Waiting
                      </span>
                    )}
                  </div>
                </div>
                {i < UNIFIED_STAGES.length - 1 && (
                  <div className="stage-connector" style={{
                    background: isDone ? 'var(--green)' : 'var(--border)'
                  }} />
                )}
              </div>
            )
          })}
        </div>
      </div>
    </>
  )
}
