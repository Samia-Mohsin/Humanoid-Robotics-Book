// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics Educational Platform',
  tagline: 'An interactive educational platform for learning Physical AI and Humanoid Robotics with multilingual support',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://humanoid-robotics-book.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'humanoid-robotics-book', // Usually your GitHub org/user name.
  projectName: 'humanoid-robotics-book.github.io', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur'], // Adding Urdu locale for localization support
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/humanoid-robotics-book/humanoid-robotics-book/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/humanoid-robotics-book/humanoid-robotics-book/tree/main/packages/create-docusaurus/templates/shared/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      colorMode: {
        defaultMode: 'light',
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Physical AI & Humanoid Robotics Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'bookSidebar',
            position: 'left',
            label: 'Book',
          },
          {
            type: 'localeDropdown',
            position: 'right',
          },
          {
            href: 'https://github.com/humanoid-robotics-book/humanoid-robotics-book',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Documentation',
            items: [
              {
                label: 'Getting Started',
                to: '/docs/intro',
              },
              {
                label: 'Chapter 1',
                to: '/docs/chapter1',
              },
              {
                label: 'Module 1: ROS2',
                to: '/docs/module-1-ros2/chapter-1-architecture',
              },
              {
                label: 'Module 2: Simulation',
                to: '/docs/module-2-simulation/chapter-1-gazebo-setup',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/humanoid-robotics-book/humanoid-robotics-book',
              },
              {
                label: 'Discord',
                href: 'https://discord.gg/humanoid-robotics',
              },
              {
                label: 'Research Papers',
                href: 'https://humanoid-robotics-book.github.io/research',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'Capstone Project',
                to: '/docs/capstone-project/intro',
              },
              {
                label: 'Module 3: AI Brain',
                to: '/docs/module-3-ai-brain/chapter-1-isaac-sim-setup',
              },
              {
                label: 'Module 4: VLA Systems',
                to: '/docs/module-4-vla/chapter-1-vla-overview',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Educational Platform. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash', 'json', 'yaml', 'docker', 'powershell'],
      },
      algolia: {
        // The application ID provided by Algolia
        appId: 'YOUR_ALGOLIA_APP_ID',
        // Public API key: it is safe to commit it
        apiKey: 'YOUR_ALGOLIA_API_KEY',
        indexName: 'humanoid-robotics-book',
        contextualSearch: true,
      },
    }),
};

export default config;