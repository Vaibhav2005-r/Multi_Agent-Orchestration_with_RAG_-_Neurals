import { Search, Shield, Layers, Cpu } from 'lucide-react'

const STAGE_META = {
  query_processing: {
    label: 'Query Processing',
    icon: <Search size={14} />,
    color: 'var(--cyan)'
  },
  security: {
    label: 'Security Layer',
    icon: <Shield size={14} />,
    color: 'var(--yellow)'
  },
  rag_retrieval: {
    label: 'RAG Retrieval',
    icon: <Layers size={14} />,
    color: 'var(--purple)'
  },
  answer_synthesis: {
    label: 'Answer Synthesis',
    icon: <Cpu size={14} />,
    color: 'var(--green)'
  }
}

function renderDetail(key, stages) {
  if (key === 'query_processing') {
    const s = stages.query_processing
    return (
      <div>
        <div className="psr-detail" style={{ marginBottom: 6 }}>
          <strong>Intent:</strong>{' '}
          <span className="psr-tag">{s.intent}</span>
        </div>
        {s.entities?.length > 0 && (
          <div className="psr-detail" style={{ marginBottom: 6 }}>
            <strong>Entities:</strong>{' '}
            {s.entities.map((e, i) => <span key={i} className="psr-tag">{e}</span>)}
          </div>
        )}
        {s.cleaned_query !== s.original_query && (
          <div className="psr-detail">
            <strong>Cleaned:</strong> {s.cleaned_query}
          </div>
        )}
        {s.enriched_query && s.enriched_query !== s.cleaned_query && (
          <div className="psr-detail" style={{ marginTop: 4, fontStyle: 'italic', color: 'var(--text-muted)' }}>
            <strong>Enriched:</strong> {s.enriched_query}
          </div>
        )}
      </div>
    )
  }

  if (key === 'security') {
    const s = stages.security
    const isAllow = s.status === 'ALLOW'
    return (
      <div>
        <div className="psr-detail">
          Status:{' '}
          <span className={`psr-tag`} style={{
            background: isAllow ? 'var(--green-dim)' : 'var(--red-dim)',
            color: isAllow ? 'var(--green)' : 'var(--red)',
            border: `1px solid ${isAllow ? 'rgba(0,255,157,0.3)' : 'rgba(255,56,96,0.3)'}`
          }}>
            {s.status}
          </span>
        </div>
        {s.reason && (
          <div className="psr-detail" style={{ marginTop: 4, color: 'var(--red)' }}>
            Reason: {s.reason}
          </div>
        )}
        {s.safe_query && s.safe_query !== s.reason && (
          <div className="psr-detail" style={{ marginTop: 4 }}>
            Safe Query: <em>{s.safe_query}</em>
          </div>
        )}
      </div>
    )
  }

  if (key === 'rag_retrieval') {
    const s = stages.rag_retrieval
    return (
      <div>
        <div className="psr-detail">
          Status:{' '}
          <span className="psr-tag" style={{
            background: s.status === 'SUCCESS' ? 'var(--green-dim)' : 'var(--yellow-dim)',
            color: s.status === 'SUCCESS' ? 'var(--green)' : 'var(--yellow)',
            border: `1px solid ${s.status === 'SUCCESS' ? 'rgba(0,255,157,0.3)' : 'rgba(245,197,24,0.3)'}`
          }}>
            {s.status}
          </span>
          {' '}· <strong>{s.num_documents}</strong> documents retrieved
        </div>
        {s.sources?.length > 0 && (
          <div className="psr-detail" style={{ marginTop: 6 }}>
            Sources:{' '}
            {s.sources.map((src, i) => (
              <span key={i} className="psr-tag">{src}</span>
            ))}
          </div>
        )}
      </div>
    )
  }

  if (key === 'answer_synthesis') {
    const s = stages.answer_synthesis
    return (
      <div>
        <div className="psr-detail">
          Generator: <span className="psr-tag">{s.model}</span>
        </div>
        <div className="psr-detail" style={{ marginTop: 4 }}>
          Fact-Checker: <span className="psr-tag">{s.hallucination_model}</span>
        </div>
      </div>
    )
  }

  return null
}

export default function PipelineStages({ stages }) {
  if (!stages) return null
  const stageKeys = ['query_processing', 'security', 'rag_retrieval', 'answer_synthesis']

  return (
    <div className="pipeline-stages-panel">
      {stageKeys.map((key, i) => {
        const meta = STAGE_META[key]
        if (!stages[key]) return null
        return (
          <div key={key} className="pipeline-stage-row">
            <div className="psr-icon" style={{ color: meta.color, background: `${meta.color}15`, border: `1px solid ${meta.color}40` }}>
              {meta.icon}
            </div>
            <div style={{ flex: 1 }}>
              <div className="psr-title">{meta.label}</div>
              {renderDetail(key, stages)}
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', flexShrink: 0 }}>
              0{i + 1}
            </div>
          </div>
        )
      })}
    </div>
  )
}
