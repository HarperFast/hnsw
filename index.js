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
// pick the first candidate that actually LOADS (a prebuilt binary can exist but fail to
// link, e.g. built against a newer glibc than this system's — fall through to a local build)
let native0;
const failures = [];
for (const candidate of candidates) {
	if (!existsSync(candidate)) continue;
	try {
		native0 = require(candidate);
		break;
	} catch (error) {
		failures.push(`${candidate}: ${error.message}`);
	}
}
if (!native0) {
	throw new Error(
		`@harperfast/hnsw could not load a native binary for ${process.platform}-${process.arch}. ` +
			'Build one with `npm run build` in ' +
			__dirname +
			' (requires a Rust toolchain: https://rustup.rs).' +
			(failures.length ? ` Load failures: ${failures.join('; ')}` : '')
	);
}
const native = native0;

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
