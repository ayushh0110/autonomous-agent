import './index.css';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Skills from './components/Skills';
import Experience from './components/Experience';
import Projects from './components/Projects';
import Contact from './components/Contact';

export default function App() {
  return (
    <>
      {/* Animated background */}
      <div className="bg-grid" />
      <div className="bg-orb bg-orb-1" />
      <div className="bg-orb bg-orb-2" />
      <div className="bg-orb bg-orb-3" />

      <Navbar />

      <main style={{ position: 'relative', zIndex: 1 }}>
        <Hero />

        {/* About — short inline section */}
        <section id="about" className="section" style={{ paddingTop: 0 }}>
          <div className="container">
            <div className="about-grid">
              <div className="about-text">
                <p className="section-label">About Me</p>
                <h2 className="section-title" style={{ marginBottom: '24px' }}>
                  Building <span className="grad-text">intelligent</span> systems
                </h2>
                <p>
                  I'm a final-year IT Engineering graduate from Bhartiya Vidyapeeth University (CGPA: 9.05) with a deep interest in 
                  artificial intelligence, agentic systems, and production software engineering.
                </p>
                <p>
                  My work spans the full stack — from architecting multi-step agentic pipelines with custom Planner–Executor–Critic 
                  loops and hybrid RAG memory, to fine-tuning transformers and deploying ML models on cloud infrastructure.
                </p>
                <p>
                  Currently at Cognizant as a Programmer Analyst, and actively building in the AI/ML space.
                </p>
              </div>
              <div className="about-card">
                <div className="about-items">
                  {[
                    { icon: '🎓', label: 'Education', val: 'B.Tech IT, BVU Pune — 9.05 CGPA' },
                    { icon: '📍', label: 'Location', val: 'Lucknow, Uttar Pradesh, India' },
                    { icon: '💼', label: 'Current Role', val: 'Programmer Analyst @ Cognizant' },
                    { icon: '🔭', label: 'Interests', val: 'Agentic AI, LLMs, Systems Design' },
                    { icon: '📬', label: 'Email', val: 'shekharayush56@gmail.com' },
                  ].map(item => (
                    <div className="about-item" key={item.label}>
                      <div className="about-item-icon">{item.icon}</div>
                      <div className="about-item-text">
                        <div className="about-item-label">{item.label}</div>
                        <div className="about-item-val">{item.val}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        <Skills />
        <Experience />
        <Projects />
        <Contact />
      </main>

      <footer className="footer">
        <p>
          Designed & built by <span>Ayush Shekhar</span> · {new Date().getFullYear()}
        </p>
      </footer>
    </>
  );
}
