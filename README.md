# AI Native Book

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Features

- Interactive textbook content
- Documentation support in Markdown and PDF formats
- Multiple pages and sections for comprehensive content

## Installation

### Prerequisites

- Node.js (v18 or higher)

### Quick Setup

1. Clone the repository
2. Place your content in the `docs/` directory

### Manual Setup

1. Install Node.js dependencies:
```bash
yarn
```

2. In a new terminal, start the frontend:
```bash
yarn start
```

## Local Development

```bash
yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build for Production

```bash
yarn build
```

This command generates static content into the `build` directory and can be served using any static content hosting service.

## Deployment

### Vercel Deployment

1. Push your code to a GitHub repository
2. Connect your repository to Vercel
3. Vercel will automatically build and deploy your Docusaurus site
