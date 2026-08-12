import {
  FileText,
  MessageSquare,
  Database,
  Cpu,
  Activity,
  Layers
} from 'lucide-react'

const NAV_ITEMS = [
  {
    id: 'ingestion',
    label: 'Ingestion Pipeline',
    icon: <Database size={16} />,
    section: 'Pipelines'
  },
  {
    id: 'chat',
    label: 'Query & Chat',
    icon: <MessageSquare size={16} />,
    section: 'Query Interface'
  }
]

export default function Sidebar({ activePage, onNavigate }) {
  const sections = [...new Set(NAV_ITEMS.map(i => i.section))]

  return (
    <aside className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <h1>NeuralRAG</h1>
        <span>Multi-Agent System</span>
      </div>

      {/* Nav */}
      <nav className="sidebar-nav">
        {sections.map(section => (
          <div key={section}>
            <div className="nav-section-label">{section}</div>
            {NAV_ITEMS.filter(i => i.section === section).map(item => (
              <div
                key={item.id}
                className={`nav-item ${activePage === item.id ? 'active' : ''}`}
                onClick={() => onNavigate(item.id)}
              >
                {item.icon}
                <span>{item.label}</span>
              </div>
            ))}
          </div>
        ))}
      </nav>

      {/* Footer status */}
      <div className="sidebar-footer">
        <div className="status-indicator">
          <div className="status-dot" />
          <span>System Online</span>
        </div>
      </div>
    </aside>
  )
}
