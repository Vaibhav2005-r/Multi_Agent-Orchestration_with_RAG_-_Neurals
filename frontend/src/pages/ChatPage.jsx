import { useState, useRef, useEffect } from 'react'
import { Send, MessageSquare } from 'lucide-react'
import MessageBubble from '../components/Chat/MessageBubble'

const API_BASE = 'http://localhost:5000/api'

// Mock response generator for when API is unavailable
function generateMockResponse(query) {
  return {
    final_answer: `**Analysis of Query: "${query}"**

Based on the retrieved regulatory documents, here is the synthesized response:

- This system processes your query through a 4-stage pipeline ensuring accuracy and security.
- The relevant regulatory guidelines have been cross-referenced and verified.
- All information has been sourced from the indexed document corpus.

> **Note:** This is a mock response. Connect the Flask API server to get real answers from the pipeline.

---
**Sources / Citations:**
1. RBI_Guidelines.pdf
2. NBFC_Rules.pdf

**Factual Confidence Score:** 🟢 87.50%

**Suggested Follow-up:** What are the penalties for non-compliance with these regulations?`,
    blocked: false,
    elapsed_seconds: 2.4,
    stages: {
      query_processing: {
        original_query: query,
        cleaned_query: query,
        intent: 'REGULATORY_GUIDELINE',
        entities: ['regulatory', 'compliance'],
        enriched_query: query + ' in the context of financial regulatory compliance'
      },
      security: { status: 'ALLOW', safe_query: query, reason: '' },
      rag_retrieval: {
        status: 'SUCCESS',
        num_documents: 3,
        sources: ['RBI_Guidelines.pdf', 'NBFC_Rules.pdf']
      },
      answer_synthesis: {
        model: 'nvidia/nemotron-3-super-120b-a12b',
        hallucination_model: 'vectara/hallucination_evaluation_model'
      }
    }
  }
}

export default function ChatPage() {
  const [messages, setMessages] = useState([])
  const [input, setInput]       = useState('')
  const [role, setRole]         = useState('EMPLOYEE')
  const [loading, setLoading]   = useState(false)
  const [apiAvail, setApiAvail] = useState(null)
  const bottomRef = useRef(null)
  const textareaRef = useRef(null)

  // Check API availability on mount
  useEffect(() => {
    fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) })
      .then(r => r.ok ? setApiAvail(true) : setApiAvail(false))
      .catch(() => setApiAvail(false))
  }, [])

  // Auto scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const sendMessage = async () => {
    if (!input.trim() || loading) return
    const userQuery = input.trim()
    setInput('')

    // Add user message
    setMessages(prev => [...prev, {
      role: 'user',
      content: userQuery,
      timestamp: new Date().toLocaleTimeString()
    }])

    setLoading(true)

    try {
      let data
      if (apiAvail) {
        const chatHistory = messages
          .filter(m => m.role !== 'user' || true)
          .map(m => ({ role: m.role, content: m.content }))

        const res = await fetch(`${API_BASE}/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: userQuery, role, chat_history: chatHistory })
        })
        data = await res.json()
      } else {
        // Use mock response if API unavailable
        await new Promise(r => setTimeout(r, 1800))
        data = generateMockResponse(userQuery)
      }

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.final_answer,
        blocked: data.blocked,
        elapsed: data.elapsed_seconds,
        stages: data.stages,
        timestamp: new Date().toLocaleTimeString()
      }])
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `**Error:** Could not reach the pipeline. ${err.message}`,
        blocked: false,
        timestamp: new Date().toLocaleTimeString()
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const handleFollowUp = (text) => {
    setInput(text)
    textareaRef.current?.focus()
  }

  return (
    <>
      {/* Header */}
      <div className="page-header">
        <div>
          <div className="page-title">Query Interface</div>
          <div className="page-subtitle">
            4-Stage Pipeline: Query Processing → Security → RAG Retrieval → Answer Synthesis
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {apiAvail === false && (
            <span className="badge badge-yellow">Demo Mode — API Offline</span>
          )}
          {apiAvail === true && (
            <span className="badge badge-green">API Connected</span>
          )}
          <span className="badge badge-purple">Role: {role}</span>
        </div>
      </div>

      <div className="chat-page">
        {/* Messages */}
        <div className="chat-messages">
          {messages.length === 0 && !loading && (
            <div className="chat-empty">
              <div className="chat-empty-icon">
                <MessageSquare size={28} />
              </div>
              <h3>Neural RAG Interface</h3>
              <p>
                Ask any question about your financial documents.
                The full 4-stage pipeline will process, secure, retrieve, and synthesize the answer.
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <MessageBubble
              key={i}
              message={msg}
              onFollowUp={handleFollowUp}
            />
          ))}

          {loading && (
            <div className="msg-row assistant">
              <div className="thinking-dots">
                <span /><span /><span />
              </div>
              <div className="msg-meta">Processing through 4-stage pipeline…</div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input Bar */}
        <div className="chat-input-bar">
          <div className="chat-input-row">
            <select
              className="role-select"
              value={role}
              onChange={e => setRole(e.target.value)}
            >
              <option value="GUEST">GUEST</option>
              <option value="EMPLOYEE">EMPLOYEE</option>
              <option value="ADMIN">ADMIN</option>
            </select>
            <textarea
              ref={textareaRef}
              className="chat-textarea"
              placeholder="Ask about loan disbursals, NBFC compliance, RBI guidelines…"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
            />
            <button
              className="send-btn"
              onClick={sendMessage}
              disabled={loading || !input.trim()}
            >
              <Send size={16} />
            </button>
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6, textAlign: 'right' }}>
            Press Enter to send · Shift+Enter for new line
          </div>
        </div>
      </div>
    </>
  )
}
