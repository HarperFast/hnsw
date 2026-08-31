// Loads the native module: a bundled prebuild for this platform when present, else a
// locally built artifact (`npm run build`, requires a Rust toolchain). Linux prebuilds are
// glibc builds; musl systems build locally.
'use strict';
const { existsSync } = require('node:fs');
const { join } = require('node:path');

const candidates = [
	join(__dirname, 'prebuilds', `${process.platform}-${process.arch}`, 'hnsw-plane.node'),
	join(__dirname, 'hnsw-plane.node'),
];
const artifact = candidates.find(existsSync);
if (!artifact) {
	throw new Error(
		`@harperfast/hnsw has no prebuilt binary for ${process.platform}-${process.arch} and no local build. ` +
			'Build one with `npm run build` in ' +
			__dirname +
			' (requires a Rust toolchain: https://rustup.rs).'
	);
}
module.exports = require(artifact);
