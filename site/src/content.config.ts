import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';

// The documentation lives in ../docs so GitHub and the site share one source.
const docs = defineCollection({
	loader: glob({ pattern: '**/*.md', base: '../docs' }),
});

export const collections = { docs };
