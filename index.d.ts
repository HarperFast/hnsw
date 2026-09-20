/**
 * Parallel arrays, ascending by distance: hit i is (ids[i], distances[i]) with host key
 * keys.subarray(keyEnds[i - 1] ?? 0, keyEnds[i]). keys/keyEnds are empty on a plane created
 * without keyCap; a hit whose node vanished mid-query has an empty key.
 */
export interface SearchHits {
	ids: Uint32Array;
	distances: Float32Array;
	keys: Buffer;
	keyEnds: Uint32Array;
}

/**
 * A persistent HNSW graph over a memory-mapped fixed-slot file. One file per index; searches
 * run on the libuv thread pool (one N-API crossing per query) and never block the JS event
 * loop. Vectors are quantized to int8 (default) or int16 with a per-vector symmetric scale.
 * See DESIGN.md for the file format, concurrency model, and durability contract.
 */
export declare class Plane {
	/**
	 * Create a new plane file. `maxNodes` is a sparse reservation — pages materialize on write.
	 * `keyCap` (default 0, else at least 8) reserves that many inline bytes per slot for the host
	 * key passed to `insert`; longer keys spill to an overflow arena of `keyArenaBytesPerNode` bytes
	 * per node (default max(128, 4 × keyCap), at least 64; sparse, so size it for the keys that will
	 * spill). Searches return the keys, so a hit resolves without a lookup by node id.
	 *
	 * `precision` fixes the stored element width for the life of the file. 'int8' (the default)
	 * quantizes each component to max|c|/127; 'int16' to max|c|/32767 — ~256× finer, at one more
	 * byte per dimension per slot (+18% at 128 dims, +73% at 1536). Whether that is close
	 * enough to rank without reranking hits against exact distances depends on your corpus and
	 * accuracy budget — validate it against your own data. int8 stays the right choice for wide
	 * embeddings, where the doubled scan bandwidth makes traversal memory-bound.
	 * Int16 planes are written in a newer format version, so an older build of this package
	 * refuses to open one rather than misreading its slot layout.
	 */
	static create(
		path: string,
		dims: number,
		layer0Cap: number,
		maxNodes: number,
		keyCap?: number,
		keyArenaBytesPerNode?: number,
		precision?: 'int8' | 'int16'
	): Plane;
	/**
	 * Open an existing plane file. Throws on a format-version mismatch and on an invalidated
	 * plane (header latch or `.stale` sidecar): delete the file and its sidecar, rebuild.
	 */
	static open(path: string): Plane;

	readonly dims: number;
	/** The stored element codec fixed at create: 'int8' or 'int16'. */
	readonly precision: 'int8' | 'int16';
	readonly layer0Cap: number;
	/** Inline key bytes per slot (0 = the plane stores no keys). */
	readonly keyCap: number;

	/**
	 * Insert a vector; returns the allocated node id (freed ids are reused). `key` is the host's
	 * key bytes (needs a `keyCap` at create). Throws on a dimension mismatch, a full plane
	 * (maxNodes reached), an exhausted key arena, or a key the plane cannot store.
	 */
	insert(vector: Float32Array, key?: Buffer): number;
	/**
	 * Insert a chunk of records in one crossing, fanned out across `threads` worker threads
	 * inside the native module (default: every hardware thread; clamped to 4× that and to the
	 * record count) and off the event loop. `vectors` is `count × dims` row-major; `keys` and
	 * `keyEnds` (both or neither) split the concatenated key bytes the way `SearchHits` does:
	 * record i's key is `keys.subarray(keyEnds[i - 1] ?? 0, keyEnds[i])`.
	 *
	 * Resolves with every record's node id in input order. A record the plane cannot hold (a
	 * non-finite component, an over-long key) is listed in `rejected` with 0xFFFFFFFF in its
	 * `ids` slot, and the rest of the batch lands. A plane fault — full, wedged, key arena
	 * exhausted — rejects the promise once in-flight inserts have finished; the records that
	 * landed before it stay in the plane, so treat a rejected batch as a failed `insert`
	 * (the host's mapping decides what to do with the plane). Batches on one plane run one at a
	 * time; a second call waits for the first. Records land in whatever order the threads reach
	 * them, so two builds of the same input produce different, equally valid graphs.
	 *
	 * Working memory: each thread's visited set is 4 bytes per allocated id, so a batch on an
	 * 8M-node plane with 16 threads holds ~512 MB of scratch, retained in the plane's scratch
	 * pool for later batches and searches.
	 */
	insertBatch(vectors: Float32Array, keys?: Buffer, keyEnds?: Uint32Array, threads?: number): Promise<InsertBatchResult>;
	/** Delete a node; its id returns to the freelist. Pairs with insert(). */
	remove(id: number): void;

	/**
	 * Async k-NN search. `filter` is an allow-bitset over node ids (bit i of byte i>>3);
	 * filtered searches are visit-bounded by ef * filterExpansion (default 24).
	 */
	search(vector: Float32Array, k: number, ef: number, filter?: Uint8Array, filterExpansion?: number): Promise<SearchHits>;
	/**
	 * Async k-NN search with a JS predicate, evaluated in batches over a threadsafe function
	 * while traversal keeps expanding (the search thread never blocks on the event loop).
	 * The predicate returns one 0/1 byte per id. Do not await it from code the predicate
	 * itself blocks on.
	 */
	searchWithPredicate(
		vector: Float32Array,
		k: number,
		ef: number,
		predicate: (ids: Array<number>, keys: Buffer, keyEnds: Uint32Array) => Uint8Array,
		filterExpansion?: number,
		visitBudget?: number
	): Promise<SearchHits>;
	/** Synchronous search (benchmarks/tests; blocks the calling thread). */
	searchSync(vector: Float32Array, k: number, ef: number): SearchHits;

	/**
	 * Mirror a host-maintained node into the plane (dual-write mode): full node state per
	 * call, HOST-allocated id (the plane allocator is bypassed), the quantized vector plus its
	 * scale and cached 1/|v|, layer-0 neighbor ids, and per-upper-level neighbor id arrays
	 * (level 1 first). An existing upper entry is rewritten in place.
	 *
	 * `vector` is raw stored bytes in the plane's own codec: `dims` bytes on an int8 plane,
	 * and `dims` little-endian int16s — `dims × 2` bytes — on an int16 one. Int16 components
	 * must stay within ±32767; -32768 is rejected, because a pair of them overflows the SIMD
	 * kernel's accumulator lane.
	 */
	writeNodeRaw(
		id: number,
		level: number,
		vector: Buffer,
		scale: number,
		invMag: number,
		neighbors: Uint32Array,
		upper?: Array<Uint32Array> | null,
		key?: Buffer // omitted: the stored key is kept
	): void;
	/** Mark a node deleted without touching the plane freelist (dual-write mode). */
	clearNode(id: number): void;
	/**
	 * Builder-scan variant of writeNodeRaw: writes only when the slot has never been touched,
	 * so a backfill scan of an older snapshot can never overwrite newer live-mirrored state.
	 * Returns true when the scan's state was written.
	 */
	writeNodeRawIfAbsent(
		id: number,
		level: number,
		vector: Buffer,
		scale: number,
		invMag: number,
		neighbors: Uint32Array,
		upper?: Array<Uint32Array> | null,
		key?: Buffer
	): boolean;
	/**
	 * Advisory: whether the file recorded a durability barrier (flush) as its last state when
	 * opened. Crash recovery does not depend on it — a per-slot lock abandoned by a dead
	 * writer is taken over lazily at that slot by whoever waits past the takeover window.
	 */
	openedClean(): boolean;
	/** Set the graph entry point (dual-write mode mirrors the host's entry updates). */
	setEntryPoint(id: number, level: number): void;
	getEntryPoint(): Array<number>;

	/** Lifetime id high-water (allocated ids, including freed ones awaiting reuse). */
	idHighWater(): number;
	getWatermark(): number;
	setWatermark(txn: number): void;
	/**
	 * Durability barrier: flush all data, then advance the watermark (omitted = leave the
	 * stored watermark untouched), then flush the header alone — a crash between the two
	 * flushes leaves an old watermark over durable data, never a new watermark over missing
	 * data. Crash recovery is per-slot: a lock abandoned by a dead handle is detected via a
	 * kernel-owned registration (immune to pid reuse) and taken over, with the slot marked
	 * deleted until rewritten.
	 */
	flush(watermark?: number): void;
	/** flush() on the libuv thread pool — a whole-map msync can stall its calling thread. */
	flushAsync(watermark?: number): Promise<void>;
	/**
	 * In-band half of invalidateFile() only — no sidecar, so a process that cannot map the
	 * file sees nothing; prefer invalidateFile(). Sets the one-way header latch, zeroes the
	 * watermark, msyncs the header page (a 4 KB barrier, not a whole-mapping flush). From then
	 * on every handle reads watermark 0, whatever a racing flush stamps, and open() throws.
	 */
	invalidate(): void;
	/**
	 * invalidatePlane() through this handle: the in-band mark via this mapping (no second open,
	 * no second registry slot — on Windows this mapping is why the unlink failed) and the
	 * `.stale` sidecar next to the path it opened. The path must not have been replaced since.
	 */
	invalidateFile(): InvalidationOutcome;
	/** Whether the plane was invalidated, by any handle, since this one opened. */
	invalidated(): boolean;
}

/** One record `insertBatch` skipped. */
export interface BatchRejection {
	/** Position in the batch. */
	index: number;
	/** Stable fault name: 'not-finite' | 'key-unstorable' | 'dimension-mismatch'. */
	code: string;
	/** The message `insert` would have thrown for this record. */
	reason: string;
}

export interface InsertBatchResult {
	/** Node id per record, in input order; 0xFFFFFFFF where `rejected` names the record. */
	ids: Uint32Array;
	/** Ascending by index. */
	rejected: Array<BatchRejection>;
}

export interface InvalidationOutcome {
	/** The watermark was zeroed and its header page msync'd. */
	inBand: boolean;
	/** `<path>.stale` exists and is fsync'd (on POSIX, so is its directory entry). */
	sidecar: boolean;
	inBandError?: string;
	sidecarError?: string;
}

/**
 * Make a plane file that could not be deleted unadoptable, durably, through a temporary
 * handle that is unmapped and closed before this returns. Both markers are always attempted:
 * the in-band latch and the fsync'd `.stale` sidecar; open() refuses a file carrying either.
 * Throws only when neither marker became durable; nothing is deleted or renamed, and an
 * in-band mark whose msync failed may still have landed in the shared mapping (the safe
 * direction: it reads as incomplete). Idempotent. Synchronous (three small fsyncs on a cold path).
 */
export declare function invalidatePlane(path: string): InvalidationOutcome;
/** invalidatePlane() on the libuv thread pool. */
export declare function invalidatePlaneAsync(path: string): Promise<InvalidationOutcome>;
/** The sidecar convention: `<path>.stale`. */
export declare function stalePathFor(path: string): string;
