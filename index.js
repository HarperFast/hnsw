// Loads the native module. Prebuilds are a planned follow-up; until then the artifact is
// produced locally by `npm run build` (requires a Rust toolchain), and absence throws with
// a clear remedy rather than a bare MODULE_NOT_FOUND.
'use strict';
const { existsSync } = require('node:fs');
const { join } = require('node:path');

const artifact = join(__dirname, 'hnsw-plane.node');
if (!existsSync(artifact)) {
	throw new Error(
		'@harperfast/hnsw native artifact not found. Build it with `npm run build` in ' +
			__dirname +
			' (requires a Rust toolchain: https://rustup.rs).'
	);
}
module.exports = require(artifact);
