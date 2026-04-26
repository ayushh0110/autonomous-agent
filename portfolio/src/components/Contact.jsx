import { useScrollReveal } from '../hooks/useScrollReveal';

const contactLinks = [
  {
    icon: '📧',
    label: 'Email',
    value: 'shekharayush56@gmail.com',
    href: 'mailto:shekharayush56@gmail.com',
  },
  {
    icon: '💼',
    label: 'LinkedIn',
    value: 'linkedin.com/in/ayush-shekhar',
    href: 'https://linkedin.com/in/ayush-shekhar',
  },
  {
    icon: '🐙',
    label: 'GitHub',
    value: 'github.com/ayushh0110',
    href: 'https://github.com/ayushh0110',
  },
  {
    icon: '🤗',
    label: 'Hugging Face',
    value: 'huggingface.co/Ayush0110',
    href: 'https://huggingface.co/Ayush0110',
  },
];

export default function Contact() {
  const ref = useScrollReveal();

  return (
    <section id="contact" className="section" ref={ref}>
      <div className="container">
        <div className="contact-wrapper">
          <div className="visible">
            <p className="section-label">Say Hello</p>
            <h2 className="section-title">
              Let's <span className="grad-text">Connect</span>
            </h2>
            <p className="section-subtitle" style={{ marginBottom: '32px' }}>
              I'm currently open to new opportunities — whether it's a full-time role, 
              freelance project, or just a good conversation about AI. My inbox is always open.
            </p>
            <a
              href="mailto:shekharayush56@gmail.com"
              className="btn-primary"
              style={{ display: 'inline-flex', width: 'fit-content' }}
            >
              Send me an email →
            </a>
          </div>

          <div className="contact-card visible" style={{ animationDelay: '0.15s' }}>
            <p style={{ fontSize: '15px', fontWeight: '600', marginBottom: '4px' }}>Find me online</p>
            <p style={{ fontSize: '13px', color: 'var(--text-3)', marginBottom: '0' }}>
              Reach out on any platform below
            </p>
            <div className="contact-links">
              {contactLinks.map(cl => (
                <a
                  key={cl.label}
                  href={cl.href}
                  target="_blank"
                  rel="noreferrer"
                  className="contact-link-item"
                >
                  <div className="contact-link-icon">{cl.icon}</div>
                  <div>
                    <div style={{ fontSize: '12px', color: 'var(--text-3)', marginBottom: '1px' }}>{cl.label}</div>
                    <div style={{ color: 'inherit' }}>{cl.value}</div>
                  </div>
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
