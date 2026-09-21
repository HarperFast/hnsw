//! NAPI surface (feature = "napi"). One boundary crossing per operation; searches run on
//! the libuv thread pool via AsyncTask so the JS event loop is never blocked (C1).
//! The surface is deliberately Harper-agnostic — pk↔id mapping, commit-callback glue, and
//! txnlog-anchored replay live in the host application.

use crate::distance::Query;
use crate::insert::{insert_batch, insert_with_key, InsertError, InsertParams};
use crate::search::{gather_keys, search_filtered, search_predicated, PredicatePipe, PredicatedHits, SearchScratch};
use crate::graph::{KeyError, WriteError};
use crate::format::Quant;
use crate::{Graph, PlaneFile};
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi::{JsDeferred, JsFunction, JsObject, JsUnknown, NapiValue};
use napi_derive::napi;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Pooled per-query scratch (the visited array is O(nodes); never allocate per query).
struct ScratchPool(Mutex<Vec<SearchScratch>>);

impl ScratchPool {
    fn take(&self) -> SearchScratch {
        self.0.lock().unwrap().pop().unwrap_or_default()
    }
    fn put(&self, s: SearchScratch) {
        let mut pool = self.0.lock().unwrap();
        if pool.len() < 64 {
            pool.push(s);
        }
    }
}

#[napi(object)]
pub struct InvalidationOutcome {
    pub in_band: bool,
    pub sidecar: bool,
    pub in_band_error: Option<String>,
    pub sidecar_error: Option<String>,
}

impl From<crate::invalidate::Invalidation> for InvalidationOutcome {
    fn from(outcome: crate::invalidate::Invalidation) -> Self {
        InvalidationOutcome {
            in_band: outcome.in_band.is_ok(),
            sidecar: outcome.sidecar.is_ok(),
            in_band_error: outcome.in_band.err().map(|e| e.to_string()),
            sidecar_error: outcome.sidecar.err().map(|e| e.to_string()),
        }
    }
}

/// Make the plane file at `path` unadoptable, durably, through a temporary handle released
/// before this returns: the in-band latch (watermark 0 on every handle, every later open
/// refused) and the fsync'd `.stale` sidecar. Both markers are attempted; throws only when
/// neither became durable, leaving the file exactly as found. Synchronous: three small
/// fsyncs on a cold path. Use invalidatePlaneAsync where the caller can await.
#[napi]
pub fn invalidate_plane(path: String) -> Result<InvalidationOutcome> {
    crate::invalidate::invalidate_plane(std::path::Path::new(&path))
        .map(InvalidationOutcome::from)
        .map_err(|e| Error::from_reason(e.to_string()))
}

pub struct InvalidateTask {
    path: String,
}

#[napi]
impl Task for InvalidateTask {
    type Output = InvalidationOutcome;
    type JsValue = InvalidationOutcome;

    fn compute(&mut self) -> Result<Self::Output> {
        crate::invalidate::invalidate_plane(std::path::Path::new(&self.path))
            .map(InvalidationOutcome::from)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

/// invalidatePlane on the libuv thread pool, for callers that can await the fsyncs.
#[napi(ts_return_type = "Promise<InvalidationOutcome>")]
pub fn invalidate_plane_async(path: String) -> AsyncTask<InvalidateTask> {
    AsyncTask::new(InvalidateTask { path })
}

/// The sidecar convention checked at attach: `<plane path>.stale`.
#[napi]
pub fn stale_path_for(path: String) -> String {
    crate::invalidate::stale_path_for(std::path::Path::new(&path)).to_string_lossy().into_owned()
}

/// Search results as parallel typed arrays, ascending by distance: hit i is
/// (ids[i], distances[i], keys[keyEnds[i-1]..keyEnds[i]]) with keyEnds[-1] = 0. `keys` and
/// `keyEnds` are empty on a plane created without key capacity; a hit whose node vanished
/// mid-query has an empty key.
#[napi(object)]
pub struct SearchHits {
    pub ids: Uint32Array,
    pub distances: Float32Array,
    pub keys: Buffer,
    pub key_ends: Uint32Array,
}

type HitsWithKeys = (Vec<(u32, f32)>, Vec<u8>, Vec<u32>);

fn with_keys(graph: &Graph, hits: Vec<(u32, f32)>) -> HitsWithKeys {
    if graph.file.key_cap == 0 {
        return (hits, Vec::new(), Vec::new());
    }
    let ids: Vec<u32> = hits.iter().map(|&(id, _)| id).collect();
    let (keys, ends) = gather_keys(graph, &ids);
    (hits, keys, ends)
}

fn check_key(graph: &Graph, key: &[u8]) -> Result<()> {
    graph.check_key(key).map_err(|e| match e {
        KeyError::NoKeys | KeyError::TooLong => Error::from_reason(format!(
            "key of {} bytes cannot be stored (plane keyCap = {}, max 65535)",
            key.len(),
            graph.file.key_cap
        )),
    })
}

fn insert_error(error: InsertError, graph: &Graph, vector_len: usize, key_len: usize) -> Error {
    Error::from_reason(match error {
        InsertError::Full => "plane is full (maxNodes reached)".to_string(),
        InsertError::Wedged => "plane slot lock is wedged (unreclaimable holder); rebuild the index".to_string(),
        InsertError::KeyArenaFull => "plane key arena is full; rebuild the index".to_string(),
        InsertError::DimensionMismatch => {
            format!("vector has {} dims; plane was created with {}", vector_len, graph.file.dims())
        }
        InsertError::KeyUnstorable => format!(
            "key of {} bytes cannot be stored (plane keyCap = {}, max 65535)",
            key_len, graph.file.key_cap
        ),
        InsertError::NotFinite => "vector has a component that is not finite".to_string(),
    })
}

fn not_finite_error(vector: &[f32]) -> Error {
    let at = vector.iter().position(|v| !v.is_finite()).unwrap_or(0);
    Error::from_reason(format!("vector component {at} is not finite"))
}

fn record_error(error: InsertError, graph: &Graph, vector: &[f32], key: &[u8]) -> Error {
    match error {
        InsertError::NotFinite => not_finite_error(vector),
        other => insert_error(other, graph, vector.len(), key.len()),
    }
}

fn write_error(error: WriteError) -> Error {
    match error {
        WriteError::Wedged => Error::from_reason("plane slot lock is wedged (unreclaimable holder); rebuild the index"),
        WriteError::KeyArenaFull => Error::from_reason("plane key arena is full; rebuild the index"),
        WriteError::BadVector(reason) => Error::from_reason(reason),
    }
}

fn hits_to_js((hits, keys, ends): HitsWithKeys) -> SearchHits {
    let mut ids = Vec::with_capacity(hits.len());
    let mut distances = Vec::with_capacity(hits.len());
    for (id, d) in hits {
        ids.push(id);
        distances.push(d);
    }
    SearchHits {
        ids: Uint32Array::new(ids),
        distances: Float32Array::new(distances),
        keys: Buffer::from(keys),
        key_ends: Uint32Array::new(ends),
    }
}

pub struct PredicateBatch {
    ids: Vec<u32>,
    keys: Vec<u8>,
    ends: Vec<u32>,
}

fn to_unknown<T: ToNapiValue>(env: &Env, value: T) -> Result<JsUnknown> {
    unsafe { JsUnknown::from_raw(env.raw(), ToNapiValue::to_napi_value(env.raw(), value)?) }
}

pub struct SearchTask {
    graph: Arc<Graph>,
    pool: Arc<ScratchPool>,
    query: Vec<f32>,
    k: usize,
    ef: usize,
    filter: Option<Vec<u8>>,
    filter_expansion: usize,
}

#[napi]
impl Task for SearchTask {
    type Output = HitsWithKeys;
    type JsValue = SearchHits;

    fn compute(&mut self) -> Result<Self::Output> {
        let mut scratch = self.pool.take();
        let query = Query::for_plane(&self.graph.file, std::mem::take(&mut self.query));
        let (hits, _stats) = search_filtered(
            &self.graph,
            &query,
            self.k,
            self.ef,
            self.filter.as_deref(),
            self.filter_expansion,
            &mut scratch,
        );
        self.pool.put(scratch);
        Ok(with_keys(&self.graph, hits))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(hits_to_js(output))
    }
}

pub struct PredicateSearchTask {
    graph: Arc<Graph>,
    pool: Arc<ScratchPool>,
    query: Vec<f32>,
    k: usize,
    ef: usize,
    tsfn: Option<ThreadsafeFunction<PredicateBatch, ErrorStrategy::Fatal>>,
    visit_budget: u64,
}

#[napi]
impl Task for PredicateSearchTask {
    type Output = HitsWithKeys;
    type JsValue = SearchHits;

    fn compute(&mut self) -> Result<Self::Output> {
        let tsfn = self.tsfn.take().ok_or_else(|| Error::from_reason("task reused"))?;
        let (tx, rx) = std::sync::mpsc::channel::<(Vec<u32>, Vec<u8>)>();
        let mut pipe = PredicatePipe {
            dispatch: Box::new(move |ids: Vec<u32>, keys: Vec<u8>, ends: Vec<u32>| {
                let tx = tx.clone();
                let ids_echo = ids.clone();
                let status = tsfn.call_with_return_value(
                    PredicateBatch { ids, keys, ends },
                    ThreadsafeFunctionCallMode::NonBlocking,
                    move |ret: Uint8Array| {
                        // predicate errors / env teardown surface as a missing send; the
                        // drain deadline in search_predicated treats absent verdicts as deny
                        let _ = tx.send((ids_echo, ret.to_vec()));
                        Ok(())
                    },
                );
                // a closing or saturated queue drops the callback without invoking it, so this
                // batch will never answer; reporting it lets the drain finish on the batches
                // that will, instead of holding teardown for the full deadline
                status == Status::Ok
            }),
            rx,
        };
        let mut scratch = self.pool.take();
        let query = Query::for_plane(&self.graph.file, std::mem::take(&mut self.query));
        let (PredicatedHits { hits, keys, key_ends }, _stats) =
            search_predicated(&self.graph, &query, self.k, self.ef, &mut pipe, self.visit_budget, &mut scratch);
        self.pool.put(scratch);
        Ok((hits, keys, key_ends))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(hits_to_js(output))
    }
}

/// One record a batch skipped: its position in the batch, the stable fault `code`
/// (`InsertError::code`), and the message the single `insert` would have thrown.
#[napi(object)]
pub struct BatchRejection {
    pub index: u32,
    pub code: String,
    pub reason: String,
}

#[napi(object)]
pub struct BatchFailureInfo {
    pub index: u32,
    pub code: String,
    pub reason: String,
}

/// `failure` set means a plane fault stopped the batch; index.js turns that into a rejection
/// that still carries `ids` and `rejected`, which is why it is a field rather than an Err.
#[napi(object)]
pub struct InsertBatchResult {
    pub ids: Uint32Array,
    pub rejected: Vec<BatchRejection>,
    pub failure: Option<BatchFailureInfo>,
}

struct PooledScratches {
    pool: Arc<ScratchPool>,
    scratches: Vec<SearchScratch>,
}

impl PooledScratches {
    fn take(pool: &Arc<ScratchPool>, n: usize) -> Self {
        PooledScratches { pool: pool.clone(), scratches: (0..n).map(|_| pool.take()).collect() }
    }
}

impl Drop for PooledScratches {
    fn drop(&mut self) {
        for scratch in self.scratches.drain(..) {
            self.pool.put(scratch);
        }
    }
}

type BatchResolver = Box<dyn FnOnce(Env) -> Result<InsertBatchResult> + Send + 'static>;
type BatchDeferred = JsDeferred<InsertBatchResult, BatchResolver>;

/// One queued batch: inputs copied out of the JS buffers, so the worker threads never touch
/// V8 memory, plus the promise to settle.
struct BatchJob {
    graph: Arc<Graph>,
    pool: Arc<ScratchPool>,
    params: InsertParams,
    vectors: Vec<f32>,
    keys: Vec<u8>,
    key_ends: Vec<u32>,
    threads: usize,
    deferred: BatchDeferred,
}

impl BatchJob {
    fn settle(self) {
        let BatchJob { graph, pool, params, vectors, keys, key_ends, threads, deferred } = self;
        let run = || run_batch(&graph, &pool, &params, &vectors, &keys, &key_ends, threads);
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
            Ok(result) => deferred.resolve(Box::new(move |_env| Ok(result))),
            Err(_) => deferred.reject(Error::from_reason("insertBatch panicked; the plane may be inconsistent")),
        }
    }
}

/// The plane's batch thread: it runs jobs in order, exits once the queue has been idle for a
/// while (a plane that bulk-loaded once does not keep a thread for its lifetime), and is joined
/// at env teardown so no batch is still writing, or settling a promise, into a torn-down env.
struct BatchWorker {
    sender: Sender<BatchJob>,
    handle: std::thread::JoinHandle<()>,
}

type BatchWorkerSlot = Arc<Mutex<Option<BatchWorker>>>;

const BATCH_THREAD_IDLE: Duration = Duration::from_secs(5);

fn batch_worker_loop(jobs: Receiver<BatchJob>, slot: BatchWorkerSlot) {
    loop {
        match jobs.recv_timeout(BATCH_THREAD_IDLE) {
            Ok(job) => job.settle(),
            Err(RecvTimeoutError::Disconnected) => return,
            Err(RecvTimeoutError::Timeout) => {
                // retire under the slot lock, where senders enqueue, so a job sent while the
                // thread decides cannot land in a queue nobody reads
                let mut worker = slot.lock().unwrap_or_else(|p| p.into_inner());
                match jobs.try_recv() {
                    Ok(job) => {
                        drop(worker);
                        job.settle();
                    }
                    Err(_) => {
                        *worker = None;
                        return;
                    }
                }
            }
        }
    }
}

fn join_batch_worker(slot: BatchWorkerSlot) {
    let worker = slot.lock().unwrap_or_else(|p| p.into_inner()).take();
    if let Some(BatchWorker { sender, handle }) = worker {
        drop(sender);
        let _ = handle.join();
    }
}

fn run_batch(
    graph: &Graph,
    pool: &Arc<ScratchPool>,
    params: &InsertParams,
    vectors: &[f32],
    keys: &[u8],
    key_ends: &[u32],
    threads: usize,
) -> InsertBatchResult {
    let dims = graph.file.dims();
    let count = vectors.len() / dims;
    let records: Vec<(&[f32], &[u8])> = (0..count)
        .map(|i| {
            let key = if key_ends.is_empty() {
                &keys[..0]
            } else {
                let start = if i == 0 { 0 } else { key_ends[i - 1] as usize };
                &keys[start..key_ends[i] as usize]
            };
            (&vectors[i * dims..(i + 1) * dims], key)
        })
        .collect();
    let mut scratches = PooledScratches::take(pool, threads.min(count.max(1)));
    let (outcome, failure) = match insert_batch(graph, params, &records, &mut scratches.scratches) {
        Ok(outcome) => (outcome, None),
        Err(failure) => {
            let (vector, key) = records[failure.index];
            let info = BatchFailureInfo {
                index: failure.index as u32,
                code: failure.error.code().to_string(),
                reason: record_error(failure.error, graph, vector, key).reason,
            };
            (failure.outcome, Some(info))
        }
    };
    drop(scratches);
    let rejected = outcome
        .rejected
        .into_iter()
        .map(|(index, error)| {
            let (vector, key) = records[index];
            BatchRejection {
                index: index as u32,
                code: error.code().to_string(),
                reason: record_error(error, graph, vector, key).reason,
            }
        })
        .collect();
    InsertBatchResult { ids: Uint32Array::new(outcome.ids), rejected, failure }
}

pub struct FlushTask {
    graph: Arc<Graph>,
    txn: Option<u64>,
}

#[napi]
impl Task for FlushTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> Result<Self::Output> {
        self.graph.file.flush_with_watermark(self.txn).map_err(|e| Error::from_reason(e.to_string()))
    }

    fn resolve(&mut self, _env: Env, _output: Self::Output) -> Result<Self::JsValue> {
        Ok(())
    }
}

#[napi]
pub struct Plane {
    graph: Arc<Graph>,
    pool: Arc<ScratchPool>,
    params: InsertParams,
    // scratch for the synchronous insert; the Mutex keeps a misuse safe rather than fast.
    // Concurrent mutation is otherwise the crate's per-slot contract (DESIGN.md §5): a batch's
    // workers run alongside insert/remove/writeNodeRaw calls, as multi-worker hosts already do.
    insert_scratch: Mutex<SearchScratch>,
    batch_worker: BatchWorkerSlot,
    batch_cleanup_hooked: AtomicBool,
}

#[napi]
impl Plane {
    /// Create a new plane file. `maxNodes` bounds the sparse reservation (pages materialize
    /// on write). `keyCap` (default 0) is the inline key capacity per slot; longer keys spill
    /// to an overflow arena of `keyArenaBytesPerNode` bytes per node (default
    /// max(128, 4 x keyCap); sparse, so size it for the keys that will spill). `precision`
    /// fixes the stored element width for the life of the file: 'int8' (default) or 'int16',
    /// which costs one more byte per dimension per slot and quantizes ~256x finer.
    #[napi(factory)]
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        path: String,
        dims: u32,
        layer0_cap: u32,
        max_nodes: f64,
        key_cap: Option<u32>,
        key_arena_bytes_per_node: Option<u32>,
        #[napi(ts_arg_type = "'int8' | 'int16'")] precision: Option<String>,
    ) -> Result<Plane> {
        let key_cap = key_cap.unwrap_or(0) as usize;
        let quant = match precision.as_deref() {
            None | Some("int8") => Quant::Int8,
            Some("int16") => Quant::Int16,
            Some(other) => return Err(Error::from_reason(format!("unknown precision {other:?}; expected 'int8' or 'int16'"))),
        };
        let file = PlaneFile::create_with_options(
            std::path::Path::new(&path),
            dims as usize,
            layer0_cap as usize,
            max_nodes as u64,
            key_cap,
            key_arena_bytes_per_node.unwrap_or_else(|| crate::format::default_key_arena_per_node(key_cap)),
            quant,
        )
        .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Self::wrap(file))
    }

    /// Open an existing plane file (the upper-layer region lives in the same file).
    #[napi(factory)]
    pub fn open(path: String) -> Result<Plane> {
        let file = PlaneFile::open(std::path::Path::new(&path)).map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Self::wrap(file))
    }

    fn wrap(file: PlaneFile) -> Plane {
        Plane {
            graph: Arc::new(Graph::new(file)),
            pool: Arc::new(ScratchPool(Mutex::new(Vec::new()))),
            params: InsertParams::default(),
            insert_scratch: Mutex::new(SearchScratch::new()),
            batch_worker: Arc::new(Mutex::new(None)),
            batch_cleanup_hooked: AtomicBool::new(false),
        }
    }

    fn enqueue_batch(&self, env: &mut Env, job: BatchJob) -> std::result::Result<(), (BatchDeferred, Error)> {
        if !self.batch_cleanup_hooked.swap(true, Ordering::AcqRel) {
            let slot = self.batch_worker.clone();
            if let Err(error) = env.add_env_cleanup_hook(slot, join_batch_worker) {
                self.batch_cleanup_hooked.store(false, Ordering::Release);
                return Err((job.deferred, error));
            }
        }
        let mut worker = self.batch_worker.lock().unwrap_or_else(|p| p.into_inner());
        if worker.is_none() {
            let (sender, jobs) = channel::<BatchJob>();
            let slot = self.batch_worker.clone();
            let spawned = std::thread::Builder::new()
                .name("hnsw-insert-batch".into())
                .spawn(move || batch_worker_loop(jobs, slot));
            match spawned {
                Ok(handle) => *worker = Some(BatchWorker { sender, handle }),
                Err(error) => {
                    return Err((job.deferred, Error::from_reason(format!("could not start the batch thread: {error}"))))
                }
            }
        }
        worker
            .as_ref()
            .expect("worker present")
            .sender
            .send(job)
            .map_err(|failed| (failed.0.deferred, Error::from_reason("the batch thread is gone")))
    }

    /// Insert a vector; returns the allocated node id (freelist ids are reused). `key` is
    /// the host's key bytes, returned verbatim with every hit (requires a `keyCap` at create).
    /// Throws on a dimension mismatch, a full plane (maxNodes reached), an exhausted key
    /// arena, or a key the plane cannot store.
    #[napi]
    pub fn insert(&self, vector: Float32Array, key: Option<Buffer>) -> Result<u32> {
        if vector.len() != self.graph.file.dims() {
            return Err(Error::from_reason(format!(
                "vector has {} dims; plane was created with {}",
                vector.len(),
                self.graph.file.dims()
            )));
        }
        let mut scratch = self.insert_scratch.lock().unwrap();
        let key = key.as_deref().unwrap_or(&[]);
        insert_with_key(&self.graph, &vector, key, &self.params, &mut scratch)
            .map_err(|e| record_error(e, &self.graph, &vector, key))
    }

    /// Insert a chunk of records in one crossing, fanned out across `threads` worker threads
    /// inside the crate (default: hardware threads, at most 16; an explicit value is clamped to
    /// 2x the hardware threads and to the record count). `vectors` is `count x dims`
    /// row-major; `keys` and `keyEnds` (both or neither) split the concatenated key bytes as
    /// `SearchHits` does. Resolves with every record's id in input order; a record the plane
    /// cannot hold (non-finite component, unstorable key, key arena exhausted) is reported in
    /// `rejected` with 0xFFFFFFFF in its slot and the rest of the batch lands. A plane fault
    /// (full, wedged) stops the batch once in-flight inserts finish and sets `failure`, which
    /// index.js turns into a rejection still carrying `ids` and `rejected`. Batches on one
    /// plane run in order on one worker thread that retires when idle; malformed inputs reject
    /// the promise. Working memory is `threads x 4 B x idHighWater` of visited-set scratch,
    /// retained in the plane's scratch pool afterwards; the count is per plane, so a host
    /// loading several planes at once divides its cores between them.
    ///
    /// A queued or in-flight batch is not covered by `flush`'s durability promise until its
    /// promise settles: `await` every outstanding `insertBatch` before advancing the watermark,
    /// or the watermark can land over records that have not been inserted yet.
    #[napi(ts_return_type = "Promise<InsertBatchResult>")]
    pub fn insert_batch(
        &self,
        env: Env,
        vectors: Float32Array,
        keys: Option<Buffer>,
        key_ends: Option<Uint32Array>,
        threads: Option<u32>,
    ) -> Result<JsObject> {
        let mut env = env;
        let (deferred, promise) = env.create_deferred::<InsertBatchResult, BatchResolver>()?;
        match self.queue_batch(&mut env, vectors, keys, key_ends, threads, deferred) {
            Ok(()) => {}
            Err((deferred, error)) => deferred.reject(error),
        }
        Ok(promise)
    }

    fn queue_batch(
        &self,
        env: &mut Env,
        vectors: Float32Array,
        keys: Option<Buffer>,
        key_ends: Option<Uint32Array>,
        threads: Option<u32>,
        deferred: BatchDeferred,
    ) -> std::result::Result<(), (BatchDeferred, Error)> {
        let dims = self.graph.file.dims();
        if !vectors.len().is_multiple_of(dims) {
            let reason = format!("batch has {} floats, not a multiple of the plane's {} dims", vectors.len(), dims);
            return Err((deferred, Error::from_reason(reason)));
        }
        let count = vectors.len() / dims;
        let (keys, key_ends) = match (keys, key_ends) {
            (None, None) => (Vec::new(), Vec::new()),
            (Some(keys), Some(ends)) => {
                if ends.len() != count {
                    let reason = format!("keyEnds has {} entries for {} records", ends.len(), count);
                    return Err((deferred, Error::from_reason(reason)));
                }
                let mut previous = 0u32;
                for &end in ends.iter() {
                    if end < previous || end as usize > keys.len() {
                        return Err((deferred, Error::from_reason("keyEnds must be non-decreasing and end within keys")));
                    }
                    previous = end;
                }
                if previous as usize != keys.len() {
                    return Err((deferred, Error::from_reason("the last keyEnd must equal keys.length")));
                }
                (keys.to_vec(), ends.to_vec())
            }
            _ => return Err((deferred, Error::from_reason("keys and keyEnds must be given together"))),
        };
        let hardware = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
        let threads = match threads {
            None | Some(0) => hardware.min(16),
            Some(t) => (t as usize).min(hardware * 2),
        };
        let job = BatchJob {
            graph: self.graph.clone(),
            pool: self.pool.clone(),
            params: self.params,
            vectors: vectors.to_vec(),
            keys,
            key_ends,
            threads,
            deferred,
        };
        self.enqueue_batch(env, job)
    }

    /// Delete a node; its id returns to the plane freelist. Standalone-allocation mode only
    /// (pairs with insert()); dual-write hosts use clearNode instead.
    #[napi]
    pub fn remove(&self, id: u32) -> Result<()> {
        self.graph
            .delete_node(id)
            .map_err(|_| Error::from_reason("plane slot lock is wedged (unreclaimable holder); rebuild the index"))
    }

    /// Mirror a host-maintained node into the plane (dual-write phase 1): full node state
    /// per call, host-allocated id, the vector in the plane's storage codec + quantization
    /// scale + cached 1/|v|, layer-0 neighbor ids, and per-upper-level neighbor id arrays
    /// (level 1 first). An existing upper entry is rewritten in place. Idempotent per
    /// (id, state). `key` is the host's key bytes, as for `insert`; omitted, the stored key
    /// is kept.
    #[napi]
    #[allow(clippy::too_many_arguments)]
    pub fn write_node_raw(
        &self,
        id: u32,
        level: u8,
        vector: Buffer,
        scale: f64,
        inv_mag: f64,
        neighbors: Uint32Array,
        upper: Option<Vec<Uint32Array>>,
        key: Option<Buffer>,
    ) -> Result<()> {
        self.check_raw_vector(&vector)?;
        // ensure_high_water + slot_ptr have no bounds check, so a host id past the fixed
        // reservation would address past the slot region (mmap overrun) — reject it here.
        if id as u64 >= self.graph.file.max_nodes {
            return Err(Error::from_reason(format!(
                "node id {} exceeds the plane's maxNodes reservation ({})",
                id, self.graph.file.max_nodes
            )));
        }
        if !(scale as f32).is_finite() || !(inv_mag as f32).is_finite() {
            return Err(Error::from_reason("scale/invMag must be finite"));
        }
        let upper_levels: Vec<Vec<u32>> =
            upper.map(|ls| ls.iter().map(|l| l.to_vec()).collect()).unwrap_or_default();
        // reject out-of-range neighbor ids rather than letting them poison traversal
        // (SearchScratch::visit would size its array from them; distance_to skips them, but
        // a u32::MAX id costs a huge allocation before it is skipped)
        let max = self.graph.file.max_nodes;
        for &n in neighbors.iter() {
            if (n as u64) >= max {
                return Err(Error::from_reason(format!("neighbor id {n} exceeds plane capacity {max}")));
            }
        }
        for level in &upper_levels {
            for &n in level {
                if (n as u64) >= max {
                    return Err(Error::from_reason(format!("upper neighbor id {n} exceeds plane capacity {max}")));
                }
            }
        }
        let key = key.as_deref();
        check_key(&self.graph, key.unwrap_or(&[]))?;
        self.graph
            .write_node_raw_with_key(id, level, &vector, scale as f32, inv_mag as f32, &neighbors.to_vec(), &upper_levels, key)
            .map_err(write_error)
    }

    /// Builder-scan variant of writeNodeRaw: writes ONLY when the slot has never been
    /// touched (valid or deleted). A backfill scan mirroring a snapshot must not overwrite
    /// a node a concurrent live mirror already wrote with newer state — the check and the
    /// write happen under the slot's seqlock, so the race is closed across workers too.
    /// Returns true when the scan's state was written.
    #[napi]
    #[allow(clippy::too_many_arguments)]
    pub fn write_node_raw_if_absent(
        &self,
        id: u32,
        level: u8,
        vector: Buffer,
        scale: f64,
        inv_mag: f64,
        neighbors: Uint32Array,
        upper: Option<Vec<Uint32Array>>,
        key: Option<Buffer>,
    ) -> Result<bool> {
        self.check_raw_vector(&vector)?;
        if (id as u64) >= self.graph.file.max_nodes {
            return Err(Error::from_reason(format!("id {} exceeds plane capacity {}", id, self.graph.file.max_nodes)));
        }
        if !(scale as f32).is_finite() || !(inv_mag as f32).is_finite() {
            return Err(Error::from_reason("scale/invMag must be finite"));
        }
        let max = self.graph.file.max_nodes;
        for &n in neighbors.iter() {
            if (n as u64) >= max {
                return Err(Error::from_reason(format!("neighbor id {n} exceeds plane capacity {max}")));
            }
        }
        let upper_levels: Vec<Vec<u32>> =
            upper.map(|ls| ls.iter().map(|l| l.to_vec()).collect()).unwrap_or_default();
        for level_ids in &upper_levels {
            for &n in level_ids {
                if (n as u64) >= max {
                    return Err(Error::from_reason(format!("upper neighbor id {n} exceeds plane capacity {max}")));
                }
            }
        }
        let mut l0 = neighbors.to_vec();
        l0.truncate(self.graph.file.layer0_cap);
        // the untouched check and the write share one seqlock acquisition inside the crate:
        // a live mirror's newer write can never be overwritten by this scan's older snapshot
        let key = key.as_deref();
        check_key(&self.graph, key.unwrap_or(&[]))?;
        self.graph
            .write_node_if_untouched(id, level, &vector, scale as f32, inv_mag as f32, &l0, &upper_levels, key)
            .map_err(write_error)
    }

    /// Advisory: whether the file recorded a durability barrier (flush) as its last state
    /// when this handle opened it. Crash recovery does not depend on it — torn per-slot
    /// locks are taken over lazily at the affected slot.
    #[napi]
    pub fn opened_clean(&self) -> bool {
        self.graph.file.opened_clean
    }

    /// Async durability barrier on the libuv pool: same ordering contract as flush(), off
    /// the event loop — a whole-map msync over a large mapping stalls its calling thread.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn flush_async(&self, watermark: Option<f64>) -> AsyncTask<FlushTask> {
        let txn = watermark.map(|w| w as u64);
        AsyncTask::new(FlushTask { graph: self.graph.clone(), txn })
    }

    /// Mark a node deleted without touching the plane freelist (dual-write mode: the host
    /// owns id allocation).
    #[napi]
    pub fn clear_node(&self, id: u32) -> Result<()> {
        self.graph
            .clear_node(id)
            .map_err(|_| Error::from_reason("plane slot lock is wedged (unreclaimable holder); rebuild the index"))
    }

    /// Set the graph entry point (dual-write mode mirrors the host's entry-point updates).
    #[napi]
    pub fn set_entry_point(&self, id: u32, level: u32) {
        // clamp: a garbage level would make every search iterate that many empty levels
        self.graph.file.set_entry_point(id, level.min(crate::format::MAX_UPPER_LEVELS as u32));
    }

    #[napi]
    pub fn get_entry_point(&self) -> Vec<f64> {
        let (id, level) = self.graph.file.entry_point();
        vec![id as f64, level as f64]
    }

    /// Query dimensionality must match the plane: the distance kernel streams
    /// `query.len()` bytes from each slot's vector, so an oversized query would read past
    /// it into adjacent slot bytes (or off the mapping entirely).
    fn check_query_dims(&self, len: usize) -> Result<()> {
        if len != self.graph.file.dims() {
            return Err(Error::from_reason(format!(
                "query vector has {} dimensions; plane dims = {}",
                len, self.graph.file.dims()
            )));
        }
        Ok(())
    }

    #[napi(getter)]
    pub fn key_cap(&self) -> u32 {
        self.graph.file.key_cap as u32
    }

    /// Raw mirror buffers are stored bytes in the plane's codec, so a dims-length int8 buffer
    /// against an int16 plane is a mismatch, not a half-written vector.
    fn check_raw_vector(&self, vector: &[u8]) -> Result<()> {
        if vector.len() != self.graph.file.vector_bytes() {
            return Err(Error::from_reason(format!(
                "vector is {} bytes; plane is {} dims x {} ({} bytes)",
                vector.len(),
                self.graph.file.dims(),
                self.graph.file.quant().name(),
                self.graph.file.vector_bytes()
            )));
        }
        if self.graph.file.quant() == Quant::Int16 && !crate::distance::int16_bytes_in_domain(vector) {
            return Err(Error::from_reason("int16 vectors must stay within +/-32767; -32768 is not a storable value"));
        }
        Ok(())
    }

    #[napi(getter)]
    pub fn dims(&self) -> u32 {
        self.graph.file.dims() as u32
    }

    /// The stored element codec fixed at create: 'int8' or 'int16'.
    #[napi(getter)]
    pub fn precision(&self) -> String {
        self.graph.file.quant().name().to_string()
    }

    #[napi(getter)]
    pub fn layer0_cap(&self) -> u32 {
        self.graph.file.layer0_cap as u32
    }

    /// Async k-NN search on the libuv thread pool. `filter` is an optional allow-bitset
    /// over node ids (bit i of byte i>>3); filtered searches are visit-bounded by
    /// ef * filterExpansion (default 24).
    #[napi(ts_return_type = "Promise<SearchHits>")]
    pub fn search(
        &self,
        vector: Float32Array,
        k: u32,
        ef: u32,
        filter: Option<Uint8Array>,
        filter_expansion: Option<u32>,
    ) -> Result<AsyncTask<SearchTask>> {
        self.check_query_dims(vector.len())?;
        Ok(AsyncTask::new(SearchTask {
            graph: self.graph.clone(),
            pool: self.pool.clone(),
            query: vector.to_vec(),
            k: k as usize,
            ef: ef as usize,
            filter: filter.map(|f| f.to_vec()),
            filter_expansion: filter_expansion.unwrap_or(24) as usize,
        }))
    }

    /// Async k-NN search with a JS predicate: `predicate(ids: number[], keys: Buffer,
    /// keyEnds: Uint32Array) => Uint8Array` (one 0/1 byte per id, evaluated synchronously). Batches of candidate ids stream to
    /// the predicate over a ThreadsafeFunction while traversal keeps expanding — the search
    /// thread never blocks on the JS event loop until the beam itself is done, so a busy
    /// loop costs speculative overshoot (bounded by the visit budget), not latency.
    /// `visitBudget` caps layer-0 visits absolutely (a host budget may sit below ef, which a
    /// multiplier cannot express); when absent the budget is ef * filterExpansion.
    /// Must not be awaited synchronously from code the predicate itself blocks.
    #[napi(ts_return_type = "Promise<SearchHits>")]
    pub fn search_with_predicate(
        &self,
        vector: Float32Array,
        k: u32,
        ef: u32,
        #[napi(ts_arg_type = "(ids: Array<number>, keys: Buffer, keyEnds: Uint32Array) => Uint8Array")] predicate: JsFunction,
        filter_expansion: Option<u32>,
        visit_budget: Option<f64>,
    ) -> Result<AsyncTask<PredicateSearchTask>> {
        self.check_query_dims(vector.len())?;
        let tsfn: ThreadsafeFunction<PredicateBatch, ErrorStrategy::Fatal> = predicate
            .create_threadsafe_function(0, |ctx: napi::threadsafe_function::ThreadSafeCallContext<PredicateBatch>| {
                let PredicateBatch { ids, keys, ends } = ctx.value;
                let ids: Vec<f64> = ids.iter().map(|&v| v as f64).collect();
                Ok(vec![
                    to_unknown(&ctx.env, ids)?,
                    to_unknown(&ctx.env, Buffer::from(keys))?,
                    to_unknown(&ctx.env, Uint32Array::new(ends))?,
                ])
            })?;
        let ef = ef as usize;
        Ok(AsyncTask::new(PredicateSearchTask {
            graph: self.graph.clone(),
            pool: self.pool.clone(),
            query: vector.to_vec(),
            k: k as usize,
            ef,
            tsfn: Some(tsfn),
            visit_budget: visit_budget
                .map(|b| b.max(1.0) as u64)
                .unwrap_or((ef * filter_expansion.unwrap_or(24) as usize) as u64),
        }))
    }

    /// Synchronous search (benchmarks/tests; blocks the calling thread).
    #[napi]
    pub fn search_sync(&self, vector: Float32Array, k: u32, ef: u32) -> Result<SearchHits> {
        self.check_query_dims(vector.len())?;
        let mut scratch = self.pool.take();
        let query = Query::for_plane(&self.graph.file, vector.to_vec());
        let (hits, _) = search_filtered(&self.graph, &query, k as usize, ef as usize, None, 24, &mut scratch);
        self.pool.put(scratch);
        Ok(hits_to_js(with_keys(&self.graph, hits)))
    }

    /// Lifetime id high-water (allocated ids, including freed ones awaiting reuse).
    #[napi]
    pub fn id_high_water(&self) -> f64 {
        self.graph.file.id_high_water() as f64
    }

    #[napi]
    pub fn get_watermark(&self) -> f64 {
        self.graph.file.watermark() as f64
    }

    #[napi]
    pub fn set_watermark(&self, txn: f64) {
        self.graph.file.set_watermark(txn as u64);
    }

    /// Durability barrier: flush all data, then advance the watermark (defaults to the
    /// current one) and the clean-shutdown flag, then flush the header alone — so a crash
    /// between the flushes can only leave an OLD watermark over durable data (replay
    /// re-covers a suffix), never a new watermark over missing data. That promise assumes
    /// the watermark you pass already landed: `await` every outstanding `insertBatch` first,
    /// or a queued batch not yet applied is exactly a new watermark over missing data.
    #[napi]
    pub fn flush(&self, watermark: Option<f64>) -> Result<()> {
        self.graph.file.flush_with_watermark(watermark.map(|w| w as u64)).map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Durably mark this plane invalidated in band: set the one-way latch, zero the watermark,
    /// and msync the header page alone, so a host disabling a plane it cannot delete has the
    /// mark on disk before it writes any out-of-band tombstone. Synchronous by design — it is
    /// a 4 KB msync, not the whole-mapping writeback `flush` performs.
    #[napi]
    pub fn invalidate(&self) -> Result<()> {
        self.graph.file.invalidate().map_err(|e| Error::from_reason(e.to_string()))
    }

    /// invalidatePlane through this handle: the in-band mark via this mapping (no second
    /// open, no second registry slot) and the `.stale` sidecar next to the path it opened.
    #[napi]
    pub fn invalidate_file(&self) -> Result<InvalidationOutcome> {
        crate::invalidate::invalidate_file(&self.graph.file)
            .map(InvalidationOutcome::from)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Whether the plane was invalidated (by any handle) since this one opened.
    #[napi]
    pub fn invalidated(&self) -> bool {
        self.graph.file.invalidated()
    }
}
