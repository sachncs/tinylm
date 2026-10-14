// @ts-check
import { defineConfig } from 'astro/config';
import rehypeDocLinks from './rehype-links.mjs';

// The site is served from https://sachncs.github.io/tinylm/
export default defineConfig({
	site: 'https://sachncs.github.io',
	base: '/tinylm',
	trailingSlash: 'always',
	markdown: { rehypePlugins: [rehypeDocLinks] },
});
