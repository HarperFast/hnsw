/**
 * Publishes each binding in `artifacts/<target>/hnsw-plane.node` as a platform-specific
 * npm package (@harperfast/hnsw-<target>), mirroring @harperfast/rocksdb-js.
 *
 * Required environment variables:
 * - NODE_AUTH_TOKEN: npm automation token
 * - TAG: dist-tag (`latest` or `next`)
 */
import { execFileSync } from 'node:child_process';
import { copyFileSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

if (!process.env.NODE_AUTH_TOKEN) {
	throw new Error('NODE_AUTH_TOKEN environment variable is not set');
}

const __dirname = fileURLToPath(dirname(import.meta.url));
const packageJson = JSON.parse(readFileSync(resolve(__dirname, '..', 'package.json'), 'utf8'));
const tag = process.env.TAG || 'latest';
const artifactsDir = resolve(__dirname, '..', 'artifacts');
const bindingFilename = 'hnsw-plane.node';
const bindings = {};

for (const target of readdirSync(artifactsDir)) {
	const binding = join(artifactsDir, target, bindingFilename);
	try {
		if (statSync(binding).isFile()) bindings[target] = binding;
	} catch {
		// not a binding dir
	}
}

// every optionalDependency must have a matching artifact, or the publish is incomplete
for (const dep of Object.keys(packageJson.optionalDependencies)) {
	const target = dep.replace(`${packageJson.name}-`, '');
	if (!bindings[target]) throw new Error(`Binding for ${dep} not found in artifacts`);
	if (packageJson.optionalDependencies[dep] !== packageJson.version) {
		throw new Error(`${dep} is pinned to ${packageJson.optionalDependencies[dep]}, not ${packageJson.version}`);
	}
}

for (const [target, binding] of Object.entries(bindings)) {
	const [platform, arch, libc] = target.split('-');
	const packageName = `${packageJson.name}-${target}`;
	const pkgInfo = {
		name: packageName,
		version: packageJson.version,
		description: `${target} binding for ${packageJson.name}`,
		license: packageJson.license,
		repository: packageJson.repository,
		main: `./${bindingFilename}`,
		exports: { '.': `./${bindingFilename}` },
		files: [bindingFilename],
		preferUnplugged: true,
		engines: packageJson.engines,
		os: [platform],
		cpu: [arch],
		libc: libc ? [libc] : undefined,
	};
	const tmpDir = join(tmpdir(), `hnsw-${target}-${packageJson.version}`);
	mkdirSync(tmpDir, { recursive: true });
	copyFileSync(binding, join(tmpDir, bindingFilename));
	writeFileSync(join(tmpDir, 'README.md'), `# ${packageName}\n\n${target} binding for [${packageJson.name}](https://npmjs.com/package/${packageJson.name}).\n`);
	writeFileSync(join(tmpDir, 'package.json'), JSON.stringify(pkgInfo, null, 2) + '\n');
	writeFileSync(join(tmpDir, '.npmrc'), `//registry.npmjs.org/:_authToken=${process.env.NODE_AUTH_TOKEN}\n`);
	console.log(`Publishing ${packageName}@${packageJson.version} (${tag})`);
	execFileSync('npm', ['publish', '--access', 'public', '--tag', tag], { cwd: tmpDir, stdio: 'inherit' });
}
console.log('All bindings published');
