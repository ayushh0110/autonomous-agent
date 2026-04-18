import { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import AgentPanel from './AgentPanel'
import MemoryManager from '../services/memoryManager'

/* ── Slide-up ease ──────────────────────────────────────── */
const EASE_OUT = [0.25, 0.46, 0.45, 0.94]

/* ══════════════════════════════════════════════════════════
   Typing dots
   ══════════════════════════════════════════════════════════ */
function TypingIndicator() {
  return (
    <motion.div
      className="flex items-start gap-3"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -6 }}
      transition={{ duration: 0.35, ease: EASE_OUT }}
    >
      <div className="w-7 h-7 rounded-full bg-accent flex items-center justify-center
                      text-[11px] text-white shrink-0">
        ✦
      </div>
      <div className="card rounded-2xl rounded-tl-md px-4 py-3.5 flex items-center gap-1.5">
        {[0, 1, 2].map((i) => (
          <motion.span
            key={i}
            className="w-[5px] h-[5px] rounded-full bg-text-dim"
            animate={{ opacity: [0.25, 0.7, 0.25], y: [0, -2.5, 0] }}
            transition={{
              duration: 1.3,
              repeat: Infinity,
              delay: i * 0.16,
              ease: 'easeInOut',
            }}
          />
        ))}
      </div>
    </motion.div>
  )
}

/* ══════════════════════════════════════════════════════════
   Message bubble
   ══════════════════════════════════════════════════════════ */
function Message({ message, index, onViewInsights }) {
  const isUser = message.role === 'user'

  return (
    <motion.div
      className={`flex items-start gap-3 ${isUser ? 'flex-row-reverse' : ''}`}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.5,
        ease: EASE_OUT,
        delay: Math.min(index * 0.03, 0.12),
      }}
    >
      {/* Avatar */}
      <div
        className={`w-7 h-7 rounded-full flex items-center justify-center text-[11px] shrink-0 mt-0.5 ${
          isUser
            ? 'bg-surface-warm text-text-muted'
            : 'bg-accent text-white'
        }`}
      >
        {isUser ? '›' : '✦'}
      </div>

      {/* Content column */}
      <div className={`flex flex-col gap-1.5 ${isUser ? 'items-end' : 'items-start'} max-w-[75%]`}>
        <div
          className={`rounded-3xl px-6 py-4 text-[14px] leading-relaxed ${
            isUser
              ? 'bubble-user text-text-primary rounded-tr-lg'
              : 'bubble-agent text-text-primary rounded-tl-lg'
          }`}
        >
          <div className="message-content whitespace-pre-wrap">
            {message.content}
          </div>
        </div>

        {/* Insights link */}
        {!isUser && message.metadata && (
          <motion.button
            onClick={() => onViewInsights(message.metadata)}
            className="flex items-center gap-1.5 px-2 py-0.5 rounded-md
                       text-[10px] text-text-dim hover:text-text-muted
                       transition-colors duration-250 group"
            whileHover={{ x: 1.5 }}
          >
            <span className="w-[5px] h-[5px] rounded-full bg-accent opacity-50
                             group-hover:opacity-100 transition-opacity duration-300" />
            insights
            {message.metadata.confidence && (
              <span
                className={`w-[5px] h-[5px] rounded-full ${
                  message.metadata.confidence === 'high'
                    ? 'bg-emerald'
                    : message.metadata.confidence === 'medium'
                      ? 'bg-amber'
                      : 'bg-rose'
                }`}
              />
            )}
          </motion.button>
        )}
      </div>
    </motion.div>
  )
}

/* ══════════════════════════════════════════════════════════
   Empty state
   ══════════════════════════════════════════════════════════ */
function EmptyState({ onPromptClick }) {
  const prompts = [
    'Latest AI breakthroughs',
    'Explain quantum computing',
    'Compare React vs Vue',
  ]

  return (
    <motion.div
      className="flex flex-col items-center justify-center h-full gap-6 pb-20"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.6, ease: EASE_OUT }}
    >
      {/* Logo */}
      <motion.div
        className="w-14 h-14 rounded-2xl bg-accent flex items-center justify-center
                   text-xl text-white shadow-lg shadow-accent/10"
        animate={{ y: [0, -5, 0] }}
        transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
      >
        ✦
      </motion.div>

      <div className="text-center space-y-2">
        <h2 className="text-lg font-semibold text-text-primary tracking-tight">
          What can I help you with?
        </h2>
        <p className="text-[13px] text-text-muted max-w-xs leading-relaxed">
          Search the web, reason through topics,
          and build on past conversations.
        </p>
      </div>

      {/* Quick prompts */}
      <div className="flex flex-wrap justify-center gap-3 mt-2 max-w-lg">
        {prompts.map((prompt) => (
          <motion.button
            key={prompt}
            className="card rounded-xl px-5 py-2.5 text-[12.5px] text-text-muted
                       hover:text-text-secondary hover:shadow-md
                       transition-all duration-300 whitespace-nowrap"
            whileHover={{ y: -2, scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => onPromptClick(prompt)}
          >
            {prompt}
          </motion.button>
        ))}
      </div>
    </motion.div>
  )
}

/* ══════════════════════════════════════════════════════════
   Chat UI — main component
   ══════════════════════════════════════════════════════════ */
export default function ChatUI() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [panelOpen, setPanelOpen] = useState(false)
  const [panelMeta, setPanelMeta] = useState(null)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const handleViewInsights = useCallback((metadata) => {
    setPanelMeta(metadata)
    setPanelOpen(true)
  }, [])

  const handlePromptClick = useCallback((prompt) => {
    setInput(prompt)
    inputRef.current?.focus()
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    const query = input.trim()
    if (!query || isLoading) return

    setMessages((prev) => [...prev, { role: 'user', content: query }])
    setInput('')
    setIsLoading(true)

    try {
      // Inject stored memory context into the request
      const profileContext = MemoryManager.getProfileForInjection()
      const sessionContext = MemoryManager.getSessionForInjection()

      const API_BASE = import.meta.env.VITE_API_URL || '/api'
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          profile_context: profileContext.length > 0 ? profileContext : null,
          session_context: sessionContext.length > 0 ? sessionContext : null,
        }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()

      // Process memory extraction from the response
      if (data.memory_extraction) {
        MemoryManager.processExtraction(data.memory_extraction)
      }

      setMessages((prev) => [
        ...prev,
        {
          role: 'agent',
          content: data.response,
          metadata: {
            source: data.source,
            tools_used: data.tools_used,
            steps_taken: data.steps_taken,
            plan: data.plan,
            confidence: data.confidence,
            refinements: data.refinements,
            memory_used: data.memory_used,
            memory_hits: data.memory_hits,
            decision: data.decision,
            llm_calls: data.llm_calls,
            steps_skipped: data.steps_skipped,
            early_stopped: data.early_stopped,
            cache_hits: data.cache_hits,
          },
        },
      ])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'agent',
          content: `Something went wrong — ${err.message}. Make sure the backend is running.`,
          metadata: null,
        },
      ])
    } finally {
      setIsLoading(false)
      inputRef.current?.focus()
    }
  }

  const isEmpty = messages.length === 0 && !isLoading

  return (
    <>
      <div className="relative w-full flex flex-col h-dvh max-w-2xl mx-auto" style={{ zIndex: 10 }}>

        {/* ── Header ── */}
        <header className="header-glass flex items-center justify-between px-6 py-4 shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-accent flex items-center justify-center
                          text-sm text-white shadow-sm shadow-accent/15">
              ✦
            </div>
            <div>
              <h1 className="text-[14px] font-semibold text-text-primary tracking-tight">
                Agent
              </h1>
              <p className="text-[11px] text-text-dim leading-none mt-0.5">
                Autonomous assistant
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <motion.div
              className="w-[6px] h-[6px] rounded-full bg-emerald"
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
            />
            <span className="text-[11px] text-text-dim">Online</span>
          </div>
        </header>

        {/* ── Messages ── */}
        <div className="flex-1 overflow-y-auto px-6 py-6">
          <AnimatePresence>
            {isEmpty && <EmptyState onPromptClick={handlePromptClick} />}
          </AnimatePresence>

          <div className="space-y-5">
            {messages.map((msg, i) => (
              <Message
                key={i}
                message={msg}
                index={i}
                onViewInsights={handleViewInsights}
              />
            ))}
          </div>

          <AnimatePresence>
            {isLoading && (
              <div className="mt-5">
                <TypingIndicator />
              </div>
            )}
          </AnimatePresence>

          <div ref={messagesEndRef} />
        </div>

        {/* ── Input area ── */}
        <div className="shrink-0 px-6 pb-5 pt-2">
          <form onSubmit={handleSubmit}>
            <div className="input-glass rounded-2xl flex items-center gap-3 px-5 py-2.5">
              <input
                ref={inputRef}
                id="chat-input"
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask anything…"
                disabled={isLoading}
                className="flex-1 bg-transparent text-[14px] text-text-primary
                           placeholder:text-text-ghost outline-none py-2.5
                           disabled:opacity-40"
                autoComplete="off"
              />

              <motion.button
                type="submit"
                disabled={!input.trim() || isLoading}
                className="w-9 h-9 rounded-xl bg-accent flex items-center justify-center
                           text-white disabled:opacity-20 disabled:cursor-not-allowed
                           transition-opacity duration-200 shadow-sm shadow-accent/15"
                whileHover={{ scale: 1.06 }}
                whileTap={{ scale: 0.92 }}
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
                     stroke="currentColor" strokeWidth="2.5"
                     strokeLinecap="round" strokeLinejoin="round">
                  <line x1="12" y1="19" x2="12" y2="5" />
                  <polyline points="5 12 12 5 19 12" />
                </svg>
              </motion.button>
            </div>
          </form>

          <p className="text-center text-[10px] text-text-ghost mt-3 tracking-wide">
            Groq · llama-3.1-8b-instant
          </p>
        </div>
      </div>

      <AgentPanel
        isOpen={panelOpen}
        onClose={() => setPanelOpen(false)}
        metadata={panelMeta}
      />
    </>
  )
}
