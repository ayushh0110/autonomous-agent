import { useScrollReveal } from '../hooks/useScrollReveal';

const experience = [
  {
    company: 'Cognizant Technology Solutions',
    role: 'Programmer Analyst Trainee',
    date: 'Oct 2025 – Present',
    icon: '🏢',
    bullets: [
      'Quality validation and release readiness for Abbott Healthcare applications in regulated Life Sciences / HIPAA environments.',
      'Designed scenario-based test cases from functional requirements, proactively identifying edge cases in critical healthcare workflows.',
      'Collaborated with DevOps teams on defect triage, log analysis, and environment-specific reproduction across test and release pipelines.',
    ],
  },
  {
    company: 'Cognizant Technology Solutions',
    role: 'Engineer Trainee',
    date: 'Jun 2025 – Sep 2025',
    icon: '🎓',
    bullets: [
      'Enterprise training in Java, Selenium, API testing (Postman), defect tracking (Jira), Docker, Kubernetes, and Agile/Scrum practices.',
    ],
  },
  {
    company: 'CERELABS',
    role: 'Software Development Engineer (Intern)',
    date: 'Jun 2024 – Aug 2024',
    icon: '🚀',
    bullets: [
      'Built Llamate — multi-turn GPT interaction tool with user-configurable parameters and response control.',
      'Designed the React + TypeScript frontend and FastAPI backend; deployed on cloud infrastructure for scalability and fault tolerance.',
    ],
  },
];

export default function Experience() {
  const ref = useScrollReveal();

  return (
    <section id="experience" className="section" ref={ref}>
      <div className="container">
        <p className="section-label">Where I've worked</p>
        <h2 className="section-title">Work <span className="grad-text">Experience</span></h2>
        <p className="section-subtitle">
          Real-world industry exposure across enterprise software testing and AI product development.
        </p>
        <div className="exp-timeline">
          {experience.map((exp, i) => (
            <div className="exp-item visible" key={i} style={{ animationDelay: `${i * 0.15}s` }}>
              <div className="exp-dot">{exp.icon}</div>
              <div className="exp-content">
                <div className="exp-meta">
                  <span className="exp-company">{exp.company}</span>
                  <span className="exp-date">{exp.date}</span>
                </div>
                <div className="exp-role">{exp.role}</div>
                <div className="exp-bullets">
                  {exp.bullets.map((b, j) => (
                    <div className="exp-bullet" key={j}>{b}</div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
