/**
 * memoryManager.js — Profile (localStorage) + Session (sessionStorage) handler.
 *
 * Profile: Stable user facts (name, job, preferences). Max 15 entries.
 *   - Confidence-based conflict resolution (no blind overwrite)
 *   - Identity keys (name, job, role) are never deleted, only updated with higher confidence
 *   - Dedup by key
 *
 * Session: Temporary context (references, tasks). Max 10 entries, FIFO.
 *   - Only 'reference' and 'task' intents stored
 *   - Only last 5 injected into requests
 */

const PROFILE_STORAGE_KEY = 'agent_profile_memory'
const SESSION_STORAGE_KEY = 'agent_session_memory'

const MAX_PROFILE_ENTRIES = 15
const MAX_SESSION_ENTRIES = 10
const MAX_SESSION_INJECTION = 5

const IDENTITY_KEYS = new Set(['name', 'job', 'role', 'profession', 'location'])
const CONFIDENCE_TOLERANCE = 0.1 // within this range = "uncertain", keep both

// ── Profile Memory (localStorage) ─────────────────────────────────────

function getProfile() {
  try {
    const raw = localStorage.getItem(PROFILE_STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function setProfile(entries) {
  try {
    localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify(entries))
  } catch (e) {
    console.warn('[MemoryManager] Failed to write profile:', e)
  }
}

/**
 * Save a profile entry with conflict resolution.
 *
 * Rules:
 * - New confidence > existing → overwrite
 * - New confidence ≈ existing (within tolerance) and values differ → keep both (as array)
 * - New confidence < existing → discard new
 * - Identity keys are never deleted, only updated with higher confidence
 * - Max 15 entries; on overflow, evict lowest-confidence non-identity entry
 */
function saveProfile(extraction) {
  const { data, confidence, intent } = extraction
  const key = data?.key
  const value = data?.value

  if (!key || !value) return

  const entries = getProfile()
  const existingIndex = entries.findIndex((e) => e.key === key)

  if (existingIndex >= 0) {
    const existing = entries[existingIndex]
    const confDiff = confidence - existing.confidence

    if (confDiff > CONFIDENCE_TOLERANCE) {
      // New is clearly higher confidence → overwrite
      entries[existingIndex] = { key, value, confidence, intent, timestamp: Date.now() }
    } else if (confDiff >= -CONFIDENCE_TOLERANCE && existing.value !== value) {
      // Similar confidence but different values → keep both
      const combinedValue = Array.isArray(existing.value)
        ? [...new Set([...existing.value, value])]
        : existing.value === value
          ? existing.value
          : [existing.value, value]
      entries[existingIndex] = {
        key,
        value: combinedValue,
        confidence: Math.max(confidence, existing.confidence),
        intent,
        timestamp: Date.now(),
      }
    }
    // else: new confidence is lower → discard (do nothing)
  } else {
    // New key — add it
    entries.push({ key, value, confidence, intent, timestamp: Date.now() })
  }

  // Enforce cap
  while (entries.length > MAX_PROFILE_ENTRIES) {
    // Find lowest-confidence non-identity entry to evict
    let evictIndex = -1
    let lowestConf = Infinity

    for (let i = 0; i < entries.length; i++) {
      if (!IDENTITY_KEYS.has(entries[i].key) && entries[i].confidence < lowestConf) {
        lowestConf = entries[i].confidence
        evictIndex = i
      }
    }

    if (evictIndex >= 0) {
      entries.splice(evictIndex, 1)
    } else {
      // All entries are identity — evict oldest
      entries.shift()
    }
  }

  setProfile(entries)
}

// ── Session Memory (sessionStorage) ───────────────────────────────────

function getSession() {
  try {
    const raw = sessionStorage.getItem(SESSION_STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function setSession(entries) {
  try {
    sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(entries))
  } catch (e) {
    console.warn('[MemoryManager] Failed to write session:', e)
  }
}

/**
 * Save a session entry. Only 'reference' and 'task' intents accepted.
 * FIFO eviction at max 10 entries.
 */
function saveSession(extraction) {
  const { data, confidence, intent } = extraction
  const key = data?.key
  const value = data?.value

  if (!key || !value) return

  // Only store reference and task intents
  if (intent !== 'reference' && intent !== 'task') return

  const entries = getSession()

  // Dedup by key — update if exists
  const existingIndex = entries.findIndex((e) => e.key === key)
  if (existingIndex >= 0) {
    entries[existingIndex] = { key, value, confidence, intent, timestamp: Date.now() }
  } else {
    entries.push({ key, value, confidence, intent, timestamp: Date.now() })
  }

  // FIFO eviction
  while (entries.length > MAX_SESSION_ENTRIES) {
    entries.shift()
  }

  setSession(entries)
}

// ── Injection helpers ─────────────────────────────────────────────────

/**
 * Get formatted profile entries for backend injection.
 * Returns all profile entries.
 */
function getProfileForInjection() {
  return getProfile()
}

/**
 * Get formatted session entries for backend injection.
 * Returns only the last 5 entries (most recent).
 */
function getSessionForInjection() {
  const entries = getSession()
  return entries.slice(-MAX_SESSION_INJECTION)
}

// ── Main handler ──────────────────────────────────────────────────────

/**
 * Process a memory extraction result from the backend.
 * Routes to profile or session storage based on memory_type.
 */
function processExtraction(extraction) {
  if (!extraction || !extraction.store) return

  if (extraction.memory_type === 'profile') {
    saveProfile(extraction)
  } else if (extraction.memory_type === 'session') {
    saveSession(extraction)
  }
}

export const MemoryManager = {
  getProfileForInjection,
  getSessionForInjection,
  processExtraction,
  // Exposed for debugging / insights panel
  getProfile,
  getSession,
}

export default MemoryManager
