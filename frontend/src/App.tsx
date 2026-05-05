import { useState, useEffect, useRef } from 'react'
import type { KeyboardEvent } from 'react'
import './App.css'

const API = ''

type MessageRole = 'user' | 'ai' | 'error' | 'thinking'

interface Message {
  id: number
  role: MessageRole
  text: string
  model?: string
}

interface ActiveModelResponse {
  fast_model?: string
  hq_model?: string
  model?: string
}

let msgId = 0

export default function App() {
  const [fastModel, setFastModel] = useState('')
  const [hqModel, setHqModel] = useState('')
  const [hqEnabled, setHqEnabled] = useState(false)

  const [url, setUrl] = useState('')
  const [videoStatus, setVideoStatus] = useState<{ text: string; type: 'ok' | 'err' | 'info' } | null>(null)
  const [activeVideoId, setActiveVideoId] = useState<string | null>(null)

  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)

  const chatBoxRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetch(`${API}/active_model`)
      .then(r => r.json())
      .then((d: ActiveModelResponse) => {
        setFastModel((d.fast_model || d.model || '').replace('models/', ''))
        setHqModel((d.hq_model || '').replace('models/', ''))
      })
      .catch(() => setFastModel('unavailable'))
  }, [])

  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight
    }
  }, [messages])

  const activeModel = hqEnabled ? hqModel : fastModel

  async function processVideo() {
    if (!url.trim()) { setVideoStatus({ text: 'Please enter a YouTube URL.', type: 'err' }); return }

    setLoading(true)
    setVideoStatus({ text: 'Processing video…', type: 'info' })
    setActiveVideoId(null)

    try {
      const res = await fetch(`${API}/process_video`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ youtube_url: url }),
      })
      const data = await res.json()

      if (!res.ok) { setVideoStatus({ text: `Error: ${data.detail}`, type: 'err' }); return }

      setActiveVideoId(data.video_id)
      setMessages([])
      setVideoStatus({
        text: `✓ ${data.cached ? 'Loaded from cache' : 'Indexed successfully'} — Video ID: ${data.video_id}`,
        type: 'ok',
      })
    } catch {
      setVideoStatus({ text: 'Could not reach the server. Is it running?', type: 'err' })
    } finally {
      setLoading(false)
    }
  }

  async function sendQuestion() {
    const q = question.trim()
    if (!activeVideoId) {
      setMessages(m => [...m, { id: ++msgId, role: 'error', text: '⚠ Load a video first before asking questions.' }])
      return
    }
    if (!q) return

    setQuestion('')
    const thinkingId = ++msgId
    setMessages(m => [...m,
      { id: ++msgId, role: 'user', text: q },
      { id: thinkingId, role: 'thinking', text: 'Thinking…' },
    ])
    setLoading(true)

    try {
      const res = await fetch(`${API}/ask_question`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: activeVideoId, question: q, high_quality: hqEnabled }),
      })
      const data = await res.json()
      setMessages(m => m.filter(msg => msg.id !== thinkingId))

      if (!res.ok) {
        setMessages(m => [...m, { id: ++msgId, role: 'error', text: `Error: ${data.detail}` }])
        return
      }

      setMessages(m => [...m, {
        id: ++msgId,
        role: 'ai',
        text: data.answer,
        model: (data.model || '').replace('models/', ''),
      }])
    } catch {
      setMessages(m => [
        ...m.filter(msg => msg.id !== thinkingId),
        { id: ++msgId, role: 'error', text: 'Could not reach the server. Is it running?' },
      ])
    } finally {
      setLoading(false)
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter' && !e.shiftKey) sendQuestion()
  }

  return (
    <>
      <h1>YouTube <span>Q&A</span></h1>

      <div className="model-badge">
        Model: <span>{activeModel || 'loading…'}</span>
      </div>

      <div
        className={`hq-toggle-wrap${hqEnabled ? ' active' : ''}`}
        onClick={() => setHqEnabled(v => !v)}
      >
        <div className="toggle-track"><div className="toggle-thumb" /></div>
        <span>High quality mode — {hqModel || '…'}</span>
      </div>

      <div className="card">
        <h2>Video Setup</h2>
        <div className="row">
          <input
            type="text"
            value={url}
            onChange={e => setUrl(e.target.value)}
            placeholder="https://www.youtube.com/watch?v=..."
            disabled={loading}
          />
          <button onClick={processVideo} disabled={loading}>Load Video</button>
        </div>
        {videoStatus && (
          <p className={`video-status status-${videoStatus.type}`}>{videoStatus.text}</p>
        )}
      </div>

      <div className="card">
        <h2>Chat</h2>
        <div className="chat-box" ref={chatBoxRef}>
          {messages.length === 0
            ? <p className="chat-empty">Ask a question about the video to start the conversation.</p>
            : messages.map(msg => (
                <div key={msg.id} className={`bubble ${msg.role}`}>
                  {msg.text}
                  {msg.model && <span className="model-tag">{msg.model}</span>}
                </div>
              ))
          }
        </div>
        <div className="row">
          <input
            type="text"
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask a question about the video…"
            disabled={loading}
          />
          <button onClick={sendQuestion} disabled={loading}>Send</button>
        </div>
      </div>
    </>
  )
}
