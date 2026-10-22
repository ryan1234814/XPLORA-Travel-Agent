import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import './index.css'
import './App.css'
import Landing from './pages/Landing'
import App from './App.tsx'
import Login from './pages/Login'
import Signup from './pages/Signup'
import AuthGate from './components/AuthGate'
import Dashboard from './pages/Dashboard'
import { warmBackend } from './utils/apiBase'
import { refreshAuth } from './utils/auth'

warmBackend()
// Restore the session from the HTTP-only cookie before the first route renders,
// so protected pages do not bounce a signed-in user to /login.
refreshAuth()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/app" element={<App />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/dashboard" element={<AuthGate><Dashboard /></AuthGate>} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
