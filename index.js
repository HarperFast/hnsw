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
const native = require(artifact);

// A predicate that throws would surface through the ThreadsafeFunction as a fatal exception
// (aborting the process); wrap it so a throw denies the batch and rejects the search instead.
const nativeSearchWithPredicate = native.Plane.prototype.searchWithPredicate;
native.Plane.prototype.searchWithPredicate = function (vector, k, ef, predicate, ...rest) {
	let predicateError;
	const guarded = (ids) => {
		if (predicateError !== undefined) return new Uint8Array(ids.length);
		try {
			const verdicts = predicate(ids);
			return verdicts instanceof Uint8Array ? verdicts : Uint8Array.from(verdicts ?? []);
		} catch (error) {
			predicateError = error;
			return new Uint8Array(ids.length);
		}
	};
	return nativeSearchWithPredicate.call(this, vector, k, ef, guarded, ...rest).then((hits) => {
		if (predicateError !== undefined) throw predicateError;
		return hits;
	});
};

module.exports = native;
