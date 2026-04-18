import FloatingBackground from './components/FloatingBackground'
import ChatUI from './components/ChatUI'

export default function App() {
  return (
    <div className="relative w-full min-h-dvh overflow-hidden bg-surface">
      <FloatingBackground />
      <div className="relative w-full min-h-dvh flex justify-center" style={{ zIndex: 2 }}>
        <ChatUI />
      </div>
    </div>
  )
}
