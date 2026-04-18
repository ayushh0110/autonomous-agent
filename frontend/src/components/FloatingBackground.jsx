import { useEffect, useRef, useCallback } from 'react'
import { motion, useMotionValue, useSpring } from 'framer-motion'

/*
 * Cursor-following blob + floating background shapes
 *
 * 1. Cursor blob: A large, soft, low-opacity radial gradient that
 *    follows the mouse with elastic spring lag. Never snaps — always
 *    trails behind, giving a "weightless" feel.
 *
 * 2. Background shapes: 6 pastel circles with very slow sinusoidal
 *    drift + gentle parallax from mouse offset. All GPU-composited.
 */

/* ── Spring config: elastic, slow settling ──────────────── */
const CURSOR_SPRING = { damping: 25, stiffness: 50, mass: 1.5 }
const PARALLAX_SPRING = { damping: 80, stiffness: 15, mass: 3 }

/* ── Background shapes — high-visibility distinct colors ── */
const SHAPES = [
  { size: 500, x: '5%',  y: '8%',  color: 'rgba(56,189,248,0.40)',   driftDur: 30, xAmp: 25, yAmp: 18, pFactor: 0.012 },  /* sky blue */
  { size: 400, x: '74%', y: '5%',  color: 'rgba(251,146,60,0.32)',   driftDur: 36, xAmp: 20, yAmp: 30, pFactor: 0.018 },  /* warm orange */
  { size: 380, x: '58%', y: '58%', color: 'rgba(167,139,250,0.38)',  driftDur: 34, xAmp: 30, yAmp: 22, pFactor: 0.010 },  /* lavender */
  { size: 300, x: '12%', y: '65%', color: 'rgba(52,211,153,0.32)',   driftDur: 28, xAmp: 18, yAmp: 25, pFactor: 0.022 },  /* mint */
  { size: 260, x: '84%', y: '38%', color: 'rgba(251,113,133,0.30)',  driftDur: 32, xAmp: 14, yAmp: 18, pFactor: 0.015 },  /* rose */
  { size: 420, x: '38%', y: '24%', color: 'rgba(56,189,248,0.28)',   driftDur: 38, xAmp: 22, yAmp: 16, pFactor: 0.008 },  /* light cyan */
]

/* ── Single floating shape ─────────────────────────────── */
function Shape({ shape, mouseX, mouseY }) {
  const offsetX = useSpring(0, PARALLAX_SPRING)
  const offsetY = useSpring(0, PARALLAX_SPRING)

  useEffect(() => {
    const unX = mouseX.on('change', (v) => offsetX.set(v * shape.pFactor))
    const unY = mouseY.on('change', (v) => offsetY.set(v * shape.pFactor))
    return () => { unX(); unY() }
  }, [mouseX, mouseY, offsetX, offsetY, shape.pFactor])

  return (
    <motion.div
      className="absolute rounded-full pointer-events-none"
      style={{
        width: shape.size,
        height: shape.size,
        left: shape.x,
        top: shape.y,
        background: `radial-gradient(circle, ${shape.color} 0%, transparent 70%)`,
        filter: 'blur(40px)',
        x: offsetX,
        y: offsetY,
        willChange: 'transform',
      }}
      animate={{
        x: [0, shape.xAmp, -shape.xAmp * 0.5, 0],
        y: [0, -shape.yAmp, shape.yAmp * 0.7, 0],
      }}
      transition={{
        duration: shape.driftDur,
        repeat: Infinity,
        ease: 'easeInOut',
      }}
    />
  )
}

/* ── Cursor-following blob ─────────────────────────────── */
function CursorBlob({ mouseX, mouseY }) {
  const blobX = useSpring(0, CURSOR_SPRING)
  const blobY = useSpring(0, CURSOR_SPRING)

  useEffect(() => {
    const unX = mouseX.on('change', (v) => blobX.set(v))
    const unY = mouseY.on('change', (v) => blobY.set(v))
    return () => { unX(); unY() }
  }, [mouseX, mouseY, blobX, blobY])

  return (
    <motion.div
      className="fixed pointer-events-none rounded-full"
      style={{
        width: 300,
        height: 300,
        x: blobX,
        y: blobY,
        left: -150,
        top: -150,
        background: 'radial-gradient(circle, rgba(251,146,60,0.42) 0%, rgba(251,113,133,0.24) 40%, transparent 70%)',
        filter: 'blur(35px)',
        willChange: 'transform',
        zIndex: 1,
      }}
    />
  )
}

/* ── Main export ───────────────────────────────────────── */
export default function FloatingBackground() {
  const mouseX = useMotionValue(0)
  const mouseY = useMotionValue(0)
  const rafRef = useRef(null)

  const handleMouseMove = useCallback(
    (e) => {
      if (rafRef.current) return
      rafRef.current = requestAnimationFrame(() => {
        mouseX.set(e.clientX)
        mouseY.set(e.clientY)
        rafRef.current = null
      })
    },
    [mouseX, mouseY]
  )

  useEffect(() => {
    window.addEventListener('mousemove', handleMouseMove, { passive: true })
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [handleMouseMove])

  /* For background parallax, center-relative coords */
  const parallaxX = useMotionValue(0)
  const parallaxY = useMotionValue(0)

  useEffect(() => {
    const unX = mouseX.on('change', (v) => parallaxX.set(v - window.innerWidth / 2))
    const unY = mouseY.on('change', (v) => parallaxY.set(v - window.innerHeight / 2))
    return () => { unX(); unY() }
  }, [mouseX, mouseY, parallaxX, parallaxY])

  return (
    <>
      {/* Cursor-following blob */}
      <CursorBlob mouseX={mouseX} mouseY={mouseY} />

      {/* Background shapes */}
      <div
        className="fixed inset-0 overflow-hidden pointer-events-none"
        style={{ zIndex: 0 }}
        aria-hidden="true"
      >
        {/* Very subtle top vignette */}
        <div
          className="absolute inset-0"
          style={{
            background: 'radial-gradient(ellipse 80% 50% at 50% 30%, rgba(99,102,241,0.02) 0%, transparent 70%)',
          }}
        />

        {SHAPES.map((shape, i) => (
          <Shape key={i} shape={shape} mouseX={parallaxX} mouseY={parallaxY} />
        ))}
      </div>
    </>
  )
}
