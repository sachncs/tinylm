// Rewrites links between the Markdown files in ../docs into site URLs, so the
// same files read correctly on GitHub and on the site.
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DOCS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../docs');
const BASE = '/tinylm';

function visit(node, fn) {
	fn(node);
	for (const child of node.children ?? []) visit(child, fn);
}

export default function rehypeDocLinks() {
	return (tree, file) => {
		const here = path.dirname(file.path);
		visit(tree, (node) => {
			if (node.type !== 'element' || node.tagName !== 'a') return;
			const href = node.properties?.href;
			if (typeof href !== 'string' || !/\.md(#.*)?$/.test(href)) return;
			if (/^[a-z]+:/.test(href)) return;
			const [file_part, hash = ''] = href.split('#');
			const target = path.resolve(here, file_part);
			const rel = path.relative(DOCS, target);
			if (rel.startsWith('..')) return;
			const slug = rel.replace(/\.md$/, '');
			node.properties.href = `${BASE}/${slug}/${hash ? '#' + hash : ''}`;
		});
	};
}
