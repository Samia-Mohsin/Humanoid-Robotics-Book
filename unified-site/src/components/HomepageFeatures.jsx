import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Interactive Learning',
    Svg: require('../../static/img/feature-interactive.svg').default,
    description: (
      <>
        Engage with our AI-powered chatbot that answers questions about Physical AI and Humanoid Robotics.
        Select text on any page to ask specific questions about the content.
      </>
    ),
  },
  {
    title: 'Personalized Experience',
    Svg: require('../../static/img/feature-personalized.svg').default,
    description: (
      <>
        Content adapts to your background and experience level. Whether you're a beginner or expert,
        get the right level of detail and examples.
      </>
    ),
  },
  {
    title: 'Multilingual Support',
    Svg: require('../../static/img/feature-multilingual.svg').default,
    description: (
      <>
        Access content in multiple languages including Urdu. Translate any chapter with a single click
        to better understand the concepts.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} alt={title} />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}