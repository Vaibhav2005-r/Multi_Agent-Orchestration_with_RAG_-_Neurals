import { useState } from 'react'
import './App.css'
import Sidebar from './components/Sidebar'
import IngestionPage from './pages/IngestionPage'
import ChatPage from './pages/ChatPage'

function App() {
  const [activePage, setActivePage] = useState('ingestion')

  return (
    <div className="app-shell">
      <div className="grid-bg" />
      <Sidebar activePage={activePage} onNavigate={setActivePage} />
      <div className="main-content">
        {activePage === 'ingestion' ? (
          <IngestionPage />
        ) : (
          <ChatPage />
        )}
      </div>
    </div>
  )
}

export default App
