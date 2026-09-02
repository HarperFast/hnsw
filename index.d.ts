export interface SearchHit {
	id: number;
	distance: number;
}

/**
 * A persistent HNSW graph over a memory-mapped fixed-slot file. One file per index; searches
 * run on the libuv thread pool (one N-API crossing per query) and never block the JS event
 * loop. Vectors are int8-quantized with asymmetric (float query × int8 stored) cosine
 * distance. See DESIGN.md for the file format, concurrency model, and durability contract.
 */
export declare class Plane {
	/** Create a new plane file. `maxNodes` is a sparse reservation — pages materialize on write. */
	static create(path: string, dims: number, layer0Cap: number, maxNodes: number): Plane;
	/**
	 * Open an existing plane file. Throws on a format-version mismatch and on an invalidated
	 * plane (header latch or `.stale` sidecar): delete the file and its sidecar, rebuild.
	 */
	static open(path: string): Plane;

	/**
	 * Insert a vector; returns the allocated node id (freed ids are reused). Throws on a
	 * dimension mismatch or when the plane is full (maxNodes reached).
	 */
	insert(vector: Float32Array): number;
	/** Delete a node; its id returns to the freelist. Pairs with insert(). */
	remove(id: number): void;

	/**
	 * Async k-NN search. `filter` is an allow-bitset over node ids (bit i of byte i>>3);
	 * filtered searches are visit-bounded by ef * filterExpansion (default 24).
	 */
	search(vector: Float32Array, k: number, ef: number, filter?: Uint8Array, filterExpansion?: number): Promise<Array<SearchHit>>;
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
		predicate: (ids: Array<number>) => Uint8Array,
		filterExpansion?: number,
		visitBudget?: number
	): Promise<Array<SearchHit>>;
	/** Synchronous search (benchmarks/tests; blocks the calling thread). */
	searchSync(vector: Float32Array, k: number, ef: number): Array<SearchHit>;

	/**
	 * Mirror a host-maintained node into the plane (dual-write mode): full node state per
	 * call, HOST-allocated id (the plane allocator is bypassed), int8 vector bin plus
	 * quantization scale and cached 1/|v|, layer-0 neighbor ids, and per-upper-level
	 * neighbor id arrays (level 1 first). An existing upper entry is rewritten in place.
	 */
	writeNodeRaw(
		id: number,
		level: number,
		vector: Buffer,
		scale: number,
		invMag: number,
		neighbors: Uint32Array,
		upper?: Array<Uint32Array> | null
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
		upper?: Array<Uint32Array> | null
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
