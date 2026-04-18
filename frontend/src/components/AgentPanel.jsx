import { motion, AnimatePresence } from 'framer-motion'

/* ── Ease ──────────────────────────────────────────────── */
const EASE_OUT = [0.25, 0.46, 0.45, 0.94]

/* ── Badge config ─────────────────────────────────────── */
const DECISION_CONF = {
  needs_search:     { label: 'Search',  color: 'text-blue',    bg: 'bg-blue-soft',    icon: '⌕' },
  direct_answer:    { label: 'Direct',  color: 'text-emerald', bg: 'bg-emerald-soft', icon: '↯' },
  memory_sufficient:{ label: 'Memory',  color: 'text-accent',  bg: 'bg-accent-soft',  icon: '◎' },
}

const CONFIDENCE_WIDTH = { high: '100%', medium: '60%', low: '28%' }

/* ── Stat row ─────────────────────────────────────────── */
function Stat({ label, value }) {
  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-text-muted text-[11px]">{label}</span>
      <span className="text-text-secondary text-[11px] font-medium tabular-nums">{value}</span>
    </div>
  )
}

/* ── Plan steps ───────────────────────────────────────── */
function PlanTimeline({ steps }) {
  if (!steps?.length) return null
  return (
    <div>
      <p className="text-text-muted text-[10px] uppercase tracking-[0.1em] font-medium mb-3">
        Execution plan
      </p>
      <div className="relative pl-4">
        <div className="absolute left-[5px] top-1 bottom-1 w-px bg-border" />
        {steps.map((step, i) => (
          <motion.div
            key={i}
            className="relative flex items-start gap-3 pb-3 last:pb-0"
            initial={{ opacity: 0, x: -5 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.05 * i, duration: 0.3, ease: EASE_OUT }}
          >
            <div className="absolute -left-4 top-[5px] w-[6px] h-[6px] rounded-full bg-surface
                           border border-border" />
            <span className="text-text-muted text-[11px] leading-relaxed">{step}</span>
          </motion.div>
        ))}
      </div>
    </div>
  )
}

/* ── Main panel ───────────────────────────────────────── */
export default function AgentPanel({ isOpen, onClose, metadata }) {
  if (!metadata) return null

  const decision = DECISION_CONF[metadata.decision] || DECISION_CONF.needs_search
  const conf = metadata.confidence || 'medium'

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            className="fixed inset-0 bg-black/8"
            style={{ zIndex: 40 }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
          />

          {/* Panel */}
          <motion.aside
            className="fixed top-0 right-0 h-full w-[320px] bg-surface-white
                       border-l border-border flex flex-col shadow-xl shadow-black/5"
            style={{ zIndex: 50 }}
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 30, stiffness: 280 }}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-border-subtle">
              <div className="flex items-center gap-2">
                <div className="w-[6px] h-[6px] rounded-full bg-accent" />
                <h2 className="text-[13px] font-semibold text-text-primary">Insights</h2>
              </div>
              <motion.button
                onClick={onClose}
                className="w-6 h-6 rounded-md flex items-center justify-center
                           text-text-dim hover:text-text-muted hover:bg-surface-hover
                           transition-all duration-200 text-xs"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                aria-label="Close"
              >
                ✕
              </motion.button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto px-5 py-5 space-y-5">
              {/* Decision + confidence */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <div className={`px-2.5 py-1 rounded-lg ${decision.bg} flex items-center gap-1.5`}>
                    <span className="text-[11px]">{decision.icon}</span>
                    <span className={`text-[11px] font-semibold ${decision.color}`}>
                      {decision.label}
                    </span>
                  </div>
                  <span className="text-text-ghost text-[10px]">routing</span>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-text-muted text-[10px] uppercase tracking-[0.1em]">
                      Confidence
                    </span>
                    <span className="text-text-secondary text-[11px] font-medium capitalize">
                      {conf}
                    </span>
                  </div>
                  <div className="h-[3px] rounded-full bg-surface-warm overflow-hidden">
                    <motion.div
                      className="h-full rounded-full bg-accent"
                      initial={{ width: 0 }}
                      animate={{ width: CONFIDENCE_WIDTH[conf] || '50%' }}
                      transition={{ duration: 0.6, ease: EASE_OUT }}
                    />
                  </div>
                </div>
              </div>

              <div className="h-px bg-border-subtle" />

              {/* Metrics */}
              <div>
                <p className="text-text-muted text-[10px] uppercase tracking-[0.1em] font-medium mb-1">
                  Metrics
                </p>
                <div className="divide-y divide-border-subtle">
                  <Stat label="Steps taken" value={metadata.steps_taken ?? 0} />
                  <Stat label="Tools used" value={metadata.tools_used?.length ? metadata.tools_used.join(', ') : '—'} />
                  <Stat label="LLM calls" value={metadata.llm_calls ?? 0} />
                  <Stat label="Memory hits" value={metadata.memory_hits ?? 0} />
                  <Stat label="Refinements" value={metadata.refinements ?? 0} />
                  <Stat label="Cache hits" value={metadata.cache_hits ?? 0} />
                  {(metadata.steps_skipped > 0 || metadata.early_stopped) && (
                    <>
                      <Stat label="Steps skipped" value={metadata.steps_skipped ?? 0} />
                      <Stat label="Early stopped" value={metadata.early_stopped ? 'Yes' : 'No'} />
                    </>
                  )}
                </div>
              </div>

              <div className="h-px bg-border-subtle" />

              <PlanTimeline steps={metadata.plan} />
            </div>

            <div className="px-5 py-3 border-t border-border-subtle">
              <p className="text-[10px] text-text-ghost text-center">
                {metadata.source || 'planner_executor'}
              </p>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}
