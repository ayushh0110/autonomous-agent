import { useScrollReveal } from '../hooks/useScrollReveal';

const skills = [
  {
    icon: '🤖',
    title: 'AI / ML',
    tags: ['LLMs', 'Agentic AI', 'RAG', 'Vector Search', 'NLP', 'Deep Learning', 'Transformers', 'Fine-tuning'],
  },
  {
    icon: '🐍',
    title: 'Languages',
    tags: ['Python', 'Java', 'JavaScript', 'TypeScript', 'C', 'C++', 'SQL'],
  },
  {
    icon: '⚙️',
    title: 'Frameworks & APIs',
    tags: ['FastAPI', 'React', 'Flask', 'Streamlit', 'Gradio', 'Hugging Face'],
  },
  {
    icon: '☁️',
    title: 'Cloud & DevOps',
    tags: ['Docker', 'Kubernetes', 'AWS EC2', 'Vercel', 'Git', 'CI/CD'],
  },
  {
    icon: '📊',
    title: 'Data Science',
    tags: ['scikit-learn', 'TensorFlow', 'PyTorch', 'NumPy', 'Pandas', 'Matplotlib'],
  },
  {
    icon: '🛠️',
    title: 'Tools',
    tags: ['VS Code', 'PyCharm', 'Postman', 'Jira', 'Figma', 'Power BI'],
  },
];

export default function Skills() {
  const ref = useScrollReveal();

  return (
    <section id="skills" className="section" ref={ref}>
      <div className="container">
        <p className="section-label">What I work with</p>
        <h2 className="section-title">Technical <span className="grad-text">Skills</span></h2>
        <p className="section-subtitle">
          A curated stack built through real project experience — from autonomous agent pipelines to production cloud deployments.
        </p>
        <div className="skills-grid">
          {skills.map((s, i) => (
            <div className="skill-card visible" key={s.title} style={{ animationDelay: `${i * 0.08}s` }}>
              <div className="skill-card-header">
                <span className="skill-icon">{s.icon}</span>
                <span className="skill-title">{s.title}</span>
              </div>
              <div className="skill-tags">
                {s.tags.map(t => <span key={t} className="skill-tag">{t}</span>)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
