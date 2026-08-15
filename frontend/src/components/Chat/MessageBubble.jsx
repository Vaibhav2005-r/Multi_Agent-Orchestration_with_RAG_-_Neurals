import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { ChevronDown, ChevronRight, FileText, Clock, HelpCircle } from 'lucide-react'
import PipelineStages from './PipelineStages'

// Extract confidence score from final_answer string
function extractConfidence(text) {
  const match = text?.match(/(\d+\.?\d*)%/)
  if (match) return parseFloat(match[1])
  return null
}

// Extract follow-up questions from final_answer string (handles multiple separated by |)
function extractFollowUps(text) {
  const match = text?.match(/\*\*Suggested Follow-up:\*\*\s*(.+)/i)
  if (!match) return []
  return match[1]
    .split('|')
    .map(q => q.trim())
    .filter(q => q.length > 5)
}

// Extract sources from final_answer string
function extractSources(text) {
  if (!text) return []
  const section = text.match(/\*\*Sources \/ Citations:\*\*\n([\s\S]*?)(?:\n\n|\*\*|$)/)
  if (!section) return []
  return section[1]
    .split('\n')
    .filter(l => l.match(/^\d+\./))
    .map(l => l.replace(/^\d+\.\s*/, '').trim())
}

// Strip metadata sections for clean display
function stripMetadata(text) {
  if (!text) return text
  return text
    .replace(/---\n\*\*Sources \/ Citations:\*\*[\s\S]*?\n\n/, '')
    .replace(/\*\*Factual Confidence Score:\*\*.*\n\n?/, '')
    .replace(/\*\*Suggested Follow-up:\*\*.*/, '')
    .trim()
}

export default function MessageBubble({ message, onFollowUp }) {
  const [showStages, setShowStages] = useState(false)

  const isUser      = message.role === 'user'
  const isBlocked   = message.blocked
  const confidence  = extractConfidence(message.content)
  const followUps   = extractFollowUps(message.content)
  const sources     = extractSources(message.content)
  const cleanAnswer = isUser ? message.content : stripMetadata(message.content)

  const confClass = confidence >= 70 ? 'high' : confidence >= 45 ? 'medium' : 'low'
  const confEmoji = confidence >= 70 ? '🟢' : confidence >= 45 ? '🟡' : '🔴'

  return (
    <div className={`msg-row ${message.role}`}>
      {/* Bubble */}
      <div className={`msg-bubble ${message.role} ${isBlocked ? 'blocked' : ''}`}>
        {isUser ? (
          <span>{cleanAnswer}</span>
        ) : (
          <div className="md-content">
            <ReactMarkdown>{cleanAnswer}</ReactMarkdown>
          </div>
        )}
      </div>

      {/* Metadata below assistant messages */}
      {!isUser && (
        <div style={{ maxWidth: '82%', width: '100%', display: 'flex', flexDirection: 'column', gap: 8, marginTop: 4 }}>

          {/* Sources */}
          {sources.length > 0 && (
            <div className="sources-list">
              <div className="sources-title">Sources / Citations</div>
              {sources.map((src, i) => (
                <div key={i} className="source-item">
                  <FileText size={11} />
                  {src}
                </div>
              ))}
            </div>
          )}

          {/* Confidence */}
          {confidence !== null && (
            <div className="confidence-bar">
              <span style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                {confEmoji} Factual Confidence
              </span>
              <div className="confidence-track">
                <div
                  className={`confidence-fill ${confClass}`}
                  style={{ width: `${confidence}%` }}
                />
              </div>
              <span style={{ fontSize: 11, fontWeight: 600, color: confClass === 'high' ? 'var(--green)' : confClass === 'medium' ? 'var(--yellow)' : 'var(--red)', whiteSpace: 'nowrap' }}>
                {confidence.toFixed(1)}%
              </span>
            </div>
          )}

          {/* Suggested Follow-up Questions */}
          {followUps.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 2 }}>
              <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.8, color: 'var(--text-muted)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 4 }}>
                <HelpCircle size={11} color="var(--purple)" /> Suggested Follow-up Questions
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {followUps.map((q, idx) => (
                  <button
                    key={idx}
                    className="followup-chip"
                    onClick={() => onFollowUp(q)}
                    title="Click to load this question into query bar"
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 6,
                      background: 'rgba(123,47,255,0.12)',
                      border: '1px solid rgba(123,47,255,0.3)',
                      borderRadius: 16,
                      padding: '6px 12px',
                      fontSize: 12,
                      color: 'var(--text-primary)',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                      textAlign: 'left'
                    }}
                  >
                    <ChevronRight size={12} color="var(--purple)" style={{ flexShrink: 0 }} />
                    <span>{q}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Pipeline stages toggle */}
          {message.stages && (
            <div>
              <div
                className="pipeline-stages-toggle"
                onClick={() => setShowStages(s => !s)}
              >
                {showStages ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                Pipeline Details
                {message.elapsed && (
                  <span style={{ marginLeft: 'auto', fontWeight: 400, color: 'var(--text-muted)', textTransform: 'none', letterSpacing: 0 }}>
                    <Clock size={10} style={{ display: 'inline', marginRight: 3 }} />
                    {message.elapsed}s
                  </span>
                )}
              </div>
              {showStages && <PipelineStages stages={message.stages} />}
            </div>
          )}

          {/* Timestamp */}
          <div className="msg-meta">
            <span>{message.timestamp}</span>
            {isBlocked && <span className="badge badge-red">Blocked</span>}
          </div>
        </div>
      )}

      {/* User timestamp */}
      {isUser && (
        <div className="msg-meta" style={{ justifyContent: 'flex-end' }}>
          <span>{message.timestamp}</span>
        </div>
      )}
    </div>
  )
}
