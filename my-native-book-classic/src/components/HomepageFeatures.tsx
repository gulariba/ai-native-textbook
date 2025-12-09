import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

type FeatureItem = {
  title: string;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Physical AI Fundamentals',
    description: (
      <>
        Master the core concepts of Physical AI, including embodied intelligence, 
        perception-action loops, and the integration of AI with physical systems.
      </>
    ),
  },
  {
    title: 'Autonomous Agents',
    description: (
      <>
        Learn how to design and implement intelligent agents that can perceive, 
        reason, and act in physical environments.
      </>
    ),
  },
  {
    title: 'Humanoid Robotics',
    description: (
      <>
        Explore the cutting-edge field of humanoid robotics, including locomotion, 
        manipulation, and human-robot interaction.
      </>
    ),
  },
  {
    title: 'Hands-on Projects',
    description: (
      <>
        Apply your knowledge with practical projects that combine theory with 
        implementation in simulated and real environments.
      </>
    ),
  },
];

function Feature({title, description}: FeatureItem) {
  return (
    <div className={clsx('col col--3')}>
      <div className="text--center">
        <div className={styles.featureSvg}>
          <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 4.5v15m7.5-7.5h-15" />
          </svg>
        </div>
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <h2 className="text--center margin-bottom--lg">What You Will Learn</h2>
          </div>
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}