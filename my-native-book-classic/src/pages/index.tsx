import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroContent}>
          <div className={styles.heroImageContainer}>
            <div className={styles.robotWrapper}>
              <img
                src="/img/logo.svg"
                alt="AI/Humanoid Robot"
                className={clsx('hero__image', styles.robotImage)}
              />
            </div>
          </div>
          <div className={styles.heroText}>
            <h1 className="hero__title">{siteConfig.title}</h1>
            <p className="hero__subtitle">{siteConfig.tagline}</p>
            <div className={styles.buttons}>
              <Link
                className="button button--primary button--lg"
                to="/docs/chapters/chapter1-introduction">
                Start Reading
              </Link>
              <Link
                className="button button--secondary button--lg margin-left--md"
                to="/docs/chapters/chapter1-introduction">
                Explore Chapters
              </Link>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

function FeaturesSection() {
  const features = [
    {
      title: 'Physical AI',
      description: 'Embodied intelligence and perception-action loops',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="url(#gradient-physical-ai)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <defs>
            <linearGradient id="gradient-physical-ai" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#6C63FF" />
              <stop offset="100%" stopColor="#FF76E7" />
            </linearGradient>
          </defs>
          <path d="M12 2L2 7l10 5 10-5-10-5z" />
          <path d="M2 17l10 5 10-5" />
          <path d="M2 12l10 5 10-5" />
        </svg>
      )
    },
    {
      title: 'Intelligent Agents',
      description: 'Designing autonomous systems that perceive and act',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="url(#gradient-agents)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <defs>
            <linearGradient id="gradient-agents" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#3B82F6" />
              <stop offset="100%" stopColor="#6C63FF" />
            </linearGradient>
          </defs>
          <circle cx="12" cy="8" r="5" />
          <path d="M20 21a8 8 0 1 0-16 0" />
        </svg>
      )
    },
    {
      title: 'Humanoid Robotics',
      description: 'Advanced robotics with human-like interactions',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="url(#gradient-robotics)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <defs>
            <linearGradient id="gradient-robotics" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#FF76E7" />
              <stop offset="100%" stopColor="#3B82F6" />
            </linearGradient>
          </defs>
          <rect x="3" y="11" width="18" height="10" rx="2" />
          <circle cx="12" cy="6" r="1" />
          <path d="M7 11V7a5 5 0 0 1 10 0v4" />
        </svg>
      )
    },
    {
      title: 'Real-World Projects',
      description: 'Apply knowledge in practical implementations',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="url(#gradient-projects)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <defs>
            <linearGradient id="gradient-projects" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#6C63FF" />
              <stop offset="100%" stopColor="#3B82F6" />
            </linearGradient>
          </defs>
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="16" y1="13" x2="8" y2="13" />
          <line x1="16" y1="17" x2="8" y2="17" />
          <polyline points="10 9 9 9 8 9" />
        </svg>
      )
    }
  ];

  return (
    <section className={clsx(styles.features, 'features')}>
      <div className="container">
        <h2 className={styles.sectionTitle}>What You Will Learn</h2>
        <div className={styles.featuresGrid}>
          {features.map((feature, idx) => (
            <div key={idx} className={styles.featureCard}>
              <div className={styles.featureIcon}>
                {feature.icon}
              </div>
              <h3 className={styles.featureTitle}>{feature.title}</h3>
              <p className={styles.featureDescription}>{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function ChapterShowcase() {
  const chapters = [
    {
      title: 'Chapter 1: Introduction to AI-Native Development',
      description: 'Foundations of building applications with AI as a core component.'
    },
    {
      title: 'Chapter 2: Understanding Large Language Models (LLMs)',
      description: 'Architecture, capabilities, and limitations of modern language models.'
    },
    {
      title: 'Chapter 3: Building AI-Native Applications',
      description: 'Architectural patterns and integration strategies for AI applications.'
    },
    {
      title: 'Chapter 4: Tools and Frameworks for AI Development',
      description: 'Essential libraries, platforms, and development workflows.'
    },
    {
      title: 'Chapter 5: Practical Implementation Examples',
      description: 'Real-world examples of AI features in production applications.'
    },
    {
      title: 'Chapter 6: Deployment and Production Considerations',
      description: 'Strategies for deploying and scaling AI-native applications.'
    }
  ];

  return (
    <section className={clsx(styles.chapterShowcase, 'chapter-showcase')}>
      <div className="container">
        <h2>Chapter Showcase</h2>
        <div className="chapter-grid">
          {chapters.map((chapter, idx) => (
            <div key={idx} className="chapter-card">
              <h3>{chapter.title}</h3>
              <p>{chapter.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`AI Native Textbook – Physical AI & Humanoid Robotics`}
      description="A unified, AI-generated textbook for the next generation of Physical AI learners.">
      <HomepageHeader />
      <main>
        <FeaturesSection />
        <ChapterShowcase />
      </main>
    </Layout>
  );
}