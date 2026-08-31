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
	/** Open an existing plane file (format-version mismatch throws: rebuild the index). */
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
		filterExpansion?: number
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
	 * Whether the file recorded a clean shutdown when opened (create() reports true). False
	 * means torn per-slot locks were scrubbed but slot contents may be incomplete — rebuild
	 * rather than trusting the plane as a complete mirror.
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
	 * Durability barrier: flush all data, then advance the watermark (defaults to the current
	 * one) and the clean-shutdown flag, then flush the header alone — a crash between the two
	 * flushes leaves an old watermark over durable data, never a new watermark over missing
	 * data. Reopening a plane that was not cleanly flushed scrubs any torn per-slot locks.
	 */
	flush(watermark?: number): void;
}
