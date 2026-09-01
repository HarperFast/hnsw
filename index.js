// Loads the native module, mirroring @harperfast/rocksdb-js's model: a platform-specific
// optionalDependency package when one exists for this platform (linux bindings are split by
// libc), else a locally built artifact (`npm run build`, requires a Rust toolchain).
'use strict';
const { existsSync } = require('node:fs');
const { join } = require('node:path');

function libcSuffix() {
	if (process.platform !== 'linux') return '';
	let isMusl = false;
	try {
		const { glibcVersionRuntime } = process.report?.getReport?.()?.header ?? {};
		isMusl = !glibcVersionRuntime;
	} catch {
		// fall through to ldd probing
	}
	if (!isMusl) return '-glibc';
	try {
		const { execSync } = require('node:child_process');
		isMusl = execSync('ldd --version', { encoding: 'utf8', stdio: 'pipe' }).includes('musl');
	} catch {
		// ldd may not exist; keep the report-based verdict
	}
	return isMusl ? '-musl' : '-glibc';
}

const failures = [];
let native0;
// 1. platform package (published binding)
try {
	native0 = require(`@harperfast/hnsw-${process.platform}-${process.arch}${libcSuffix()}`);
} catch (error) {
	failures.push(`platform package: ${error.message}`);
}
// 2. local build (dev checkouts, source-build installs)
if (!native0) {
	const local = join(__dirname, 'hnsw-plane.node');
	if (existsSync(local)) {
		try {
			native0 = require(local);
		} catch (error) {
			failures.push(`${local}: ${error.message}`);
		}
	}
}
if (!native0) {
	throw new Error(
		`@harperfast/hnsw could not load a native binding for ${process.platform}-${process.arch}. ` +
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
