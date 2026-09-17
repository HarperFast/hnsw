// End-to-end smoke test: `npm run build && node smoke.mjs` (also the CI path).
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const { Plane, invalidatePlane, invalidatePlaneAsync, stalePathFor } = require('./index.js');

const dims = 64;
const { tmpdir } = await import('node:os');
const { join } = await import('node:path');
const path = join(tmpdir(), `smoke-${process.pid}.hnsw`);
const plane = Plane.create(path, dims, 32, 10_000, 16);
if (plane.keyCap !== 16) throw new Error('keyCap not reported');

function vec(i) {
	const v = new Float32Array(dims);
	for (let d = 0; d < dims; d++) v[d] = Math.sin(i * 0.37 + d * 1.13) * 0.1 + (d % 7 === i % 7 ? 1 : 0);
	return v;
}

// keys: short ones fit the 16-byte inline capacity, every 100th overflows to the arena
const keyFor = (i) => (i % 100 === 0 ? `record-${i}-with-a-long-key-that-overflows-the-slot` : `r${i}`);
const ids = [];
for (let i = 0; i < 2000; i++) ids.push(plane.insert(vec(i), Buffer.from(keyFor(i))));
console.log('inserted 2000, highWater =', plane.idHighWater());
const keyAt = (hits, i) => hits.keys.subarray(i === 0 ? 0 : hits.keyEnds[i - 1], hits.keyEnds[i]).toString();

// async search: nearest neighbor of an inserted vector is itself (distance ~0)
const hits = await plane.search(vec(42), 5, 128);
console.log('top hit:', hits.ids[0], hits.distances[0], keyAt(hits, 0));
if (!(hits.ids instanceof Uint32Array) || !(hits.distances instanceof Float32Array)) throw new Error('hits are not typed arrays');
if (hits.distances[0] > 1e-3) throw new Error('self-query failed');
if (hits.keyEnds.length !== hits.ids.length) throw new Error('keyEnds must parallel ids');
for (let i = 0; i < hits.ids.length; i++) {
	const idx = ids.indexOf(hits.ids[i]);
	if (keyAt(hits, i) !== keyFor(idx)) throw new Error(`key mismatch for id ${hits.ids[i]}: ${keyAt(hits, i)}`);
}
const long = await plane.search(vec(100), 3, 128);
if (![...long.ids].some((id, i) => keyAt(long, i) === keyFor(100))) throw new Error('overflow key not returned');

// filtered search: allow only even ids
const bitset = new Uint8Array(Math.ceil(plane.idHighWater() / 8));
for (const id of ids) if (id % 2 === 0) bitset[id >> 3] |= 1 << (id & 7);
const filtered = await plane.search(vec(43), 5, 128, bitset);
for (const id of filtered.ids) if (id % 2 !== 0) throw new Error(`filter leak: id ${id}`);
console.log('filtered top hit:', filtered.ids[0], filtered.distances[0]);

// delete + reinsert reuses the id (the #2182 fix)
plane.remove(ids[7]);
const reused = plane.insert(vec(9001), Buffer.from('r9001'));
if (reused !== ids[7]) throw new Error(`expected id reuse of ${ids[7]}, got ${reused}`);
console.log('freelist reuse OK, highWater still', plane.idHighWater());

// pipelined JS predicate: admit only ids divisible by 3; verdicts computed on the JS
// event loop while traversal runs on the libuv pool
let predicateCalls = 0;
const pred = await plane.searchWithPredicate(vec(44), 5, 128, (batchIds, keys, keyEnds) => {
	predicateCalls++;
	if (!(keys instanceof Uint8Array) || keyEnds.length !== batchIds.length) throw new Error('predicate batch lacks keys');
	for (let i = 0; i < batchIds.length; i++) {
		const key = keys.subarray(i === 0 ? 0 : keyEnds[i - 1], keyEnds[i]).toString();
		const expected = batchIds[i] === reused ? 'r9001' : keyFor(ids.indexOf(batchIds[i]));
		if (key !== expected) throw new Error(`predicate key mismatch for id ${batchIds[i]}: ${key} vs ${expected}`);
	}
	return Uint8Array.from(batchIds, (id) => (id % 3 === 0 ? 1 : 0));
});
for (const id of pred.ids) if (id % 3 !== 0) throw new Error(`predicate leak: id ${id}`);
if (pred.ids.length === 0) throw new Error('predicate search returned nothing');
for (let i = 0; i < pred.ids.length; i++) {
	const expected = pred.ids[i] === reused ? 'r9001' : keyFor(ids.indexOf(pred.ids[i]));
	if (keyAt(pred, i) !== expected) throw new Error(`predicated hit key mismatch for id ${pred.ids[i]}: ${keyAt(pred, i)}`);
}
console.log(`predicate top hit: id ${pred.ids[0]} (calls: ${predicateCalls})`);

// raw mirroring path (dual-write phase 1): host-allocated ids, full node state per call
const mirror = Plane.create(join(tmpdir(), `smoke-mirror-${process.pid}.hnsw`), dims, 32, 10_000);
const q42 = vec(42);
// quantize like the host: scale maps max|c| to 127, invMag = 1/|v|
function quant(v) {
	let maxAbs = 0,
		magSq = 0;
	for (const x of v) {
		maxAbs = Math.max(maxAbs, Math.abs(x));
		magSq += x * x;
	}
	const scale = maxAbs === 0 ? 1 : maxAbs / 127;
	const bytes = Buffer.from(Int8Array.from(v, (x) => Math.max(-127, Math.min(127, Math.round(x / scale)))).buffer);
	return { bytes, scale, invMag: 1 / Math.sqrt(magSq) };
}
// two nodes linked to each other, host ids 10 and 20; node 10 is the entry at level 1
const a = quant(q42),
	b = quant(vec(43));
mirror.writeNodeRaw(10, 1, a.bytes, a.scale, a.invMag, Uint32Array.from([20]), [Uint32Array.from([])]);
mirror.writeNodeRaw(20, 0, b.bytes, b.scale, b.invMag, Uint32Array.from([10]), null);
mirror.setEntryPoint(10, 1);
const mhits = mirror.searchSync(q42, 2, 16);
if (mhits.ids[0] !== 10 || mhits.distances[0] > 1e-3)
	throw new Error(`mirror self-query failed: ${JSON.stringify([...mhits.ids])}`);
mirror.clearNode(20);
const mhits2 = mirror.searchSync(vec(43), 2, 16);
if (mhits2.ids.includes(20)) throw new Error('cleared node still returned');
console.log('raw mirroring OK');

plane.flush();
const reopened = Plane.open(path);
const hits2 = reopened.searchSync(vec(42), 5, 128);
if (hits2.distances[0] > 1e-3) throw new Error('reopened self-query failed');
console.log('reopen OK');

// invalidation through the caller's own handle: both markers land, the latch survives a
// later flush, and every later open is refused
const { existsSync, mkdirSync, rmSync } = await import('node:fs');
reopened.setWatermark(4096);
reopened.flush();
const outcome = reopened.invalidateFile();
if (!outcome.inBand || !outcome.sidecar) throw new Error(`invalidation incomplete: ${JSON.stringify(outcome)}`);
if (stalePathFor(path) !== `${path}.stale` || !existsSync(stalePathFor(path))) throw new Error('no .stale sidecar');
reopened.flush(900);
if (reopened.getWatermark() !== 0 || !reopened.invalidated()) throw new Error('a later flush revived the plane');
rmSync(stalePathFor(path));
let refused;
try {
	Plane.open(path);
} catch (error) {
	refused = error;
}
if (!refused || !/invalidated/.test(refused.message)) throw new Error(`open must refuse an invalidated plane, got ${refused}`);
// by path: a temporary open that must not survive the call (idempotent on a latched plane)
const byPath = invalidatePlane(path);
if (!byPath.inBand || !byPath.sidecar) throw new Error(`path invalidation incomplete: ${JSON.stringify(byPath)}`);
const byPathAsync = await invalidatePlaneAsync(path);
if (!byPathAsync.inBand || !byPathAsync.sidecar) throw new Error(`async path invalidation incomplete: ${JSON.stringify(byPathAsync)}`);
rmSync(stalePathFor(path));
// neither marker possible: not a plane, and a directory squatting the sidecar path
const bogus = join(tmpdir(), `smoke-bogus-${process.pid}.hnsw`);
const { writeFileSync } = await import('node:fs');
writeFileSync(bogus, 'not a plane');
mkdirSync(stalePathFor(bogus));
let threw;
try {
	invalidatePlane(bogus);
} catch (error) {
	threw = error;
}
if (!threw || !/in-band:.*sidecar:/.test(threw.message)) throw new Error(`double failure must throw naming both causes, got ${threw}`);
rmSync(stalePathFor(bogus), { recursive: true });
console.log('invalidatePlane OK. smoke PASSED');
