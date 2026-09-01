// Builds the NAPI module in place (`npm run build`). With --if-needed (the npm install hook),
// a missing Rust toolchain or a failed build is a warning and exit 0 — consumers treat the
// module as optional and fall back to their own path when the artifact is absent.
import { execSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { copyFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const crateRoot = dirname(fileURLToPath(import.meta.url));
const ifNeeded = process.argv.includes('--if-needed');
const artifact = join(crateRoot, 'hnsw-plane.node');
const prebuilt = join(crateRoot, 'prebuilds', `${process.platform}-${process.arch}`, 'hnsw-plane.node');

// a prebuild satisfies --if-needed only if it actually LOADS on this system (a newer-glibc
// artifact exists but throws at require; fall through to a local build in that case)
function loads(path) {
	if (!existsSync(path)) return false;
	try {
		createRequire(import.meta.url)(path);
		return true;
	} catch {
		return false;
	}
}
if (ifNeeded && (loads(artifact) || loads(prebuilt))) {
	process.exit(0);
}
try {
	// build the lib alone: the bench bin cannot link against unresolved node-api symbols
	execSync('cargo build --release --features napi --lib', { cwd: crateRoot, stdio: 'inherit' });
} catch (error) {
	if (ifNeeded) {
		console.warn('@harperfast/hnsw: no Rust toolchain or build failed; native module unavailable until `npm run build` succeeds');
		process.exit(0);
	}
	throw error;
}
const cdylib =
	process.platform === 'win32'
		? 'hnsw_plane.dll'
		: process.platform === 'darwin'
			? 'libhnsw_plane.dylib'
			: 'libhnsw_plane.so';
copyFileSync(join(crateRoot, 'target', 'release', cdylib), artifact);
console.log('built hnsw-plane.node');
