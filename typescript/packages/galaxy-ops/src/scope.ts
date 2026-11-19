/**
 * A per-run scope for agent hosts, and the memory of which ids a run has seen as what.
 *
 * Neither is a tool parameter: a host -- an MCP service holding one session per agent run, a
 * benchmark runner -- puts both on the context it builds for that run, and the registry
 * enforces them around every operation (see operations/registry.ts, `guardScope`). A context
 * built without a scope behaves exactly as before: nothing here runs, nothing is recorded.
 *
 * This module holds the data types, the two decision functions (`historyAllowed`,
 * `handleKindsOf`), the sentences every refusal is worded from, and the helpers an op calls to
 * report what it returned (`recordHandles`, `recordRefs`). It decides nothing on its own: the
 * one place a refusal is thrown is the registry.
 */
import type { GalaxyContext } from "./context";

/** What a Galaxy id names. Galaxy's ids carry no type, so this is remembered per run. */
export type HandleKind =
  | "history"
  | "hda"
  | "hdca"
  | "job"
  | "invocation"
  | "tool_request"
  | "workflow"
  | "dataset_collection_element";

export const HANDLE_KINDS: readonly HandleKind[] = [
  "history",
  "hda",
  "hdca",
  "job",
  "invocation",
  "tool_request",
  "workflow",
  "dataset_collection_element",
];

/** How a kind reads in a sentence: "returned as a job id; dataset_id expects a dataset id". */
export const HANDLE_NOUNS: Readonly<Record<HandleKind, string>> = {
  history: "history id",
  hda: "dataset id",
  hdca: "dataset collection id",
  job: "job id",
  invocation: "invocation id",
  tool_request: "tool request id",
  workflow: "workflow id",
  dataset_collection_element: "collection element id",
};

/**
 * What one run may touch.
 *
 * Ids only, compared case-insensitively because Galaxy writes its ids in lowercase hex and
 * decodes either case. An empty `historyIds` with no read-only list is a scope that allows
 * no history at all, which is a legitimate way to run an agent that must not touch any.
 */
export interface OperationScope {
  /** Histories this run may read and write. */
  readonly historyIds: readonly string[];
  /** Histories this run may read but not change. */
  readonly readOnlyHistoryIds?: readonly string[];
  /** Operations refused by name, before anything is sent to Galaxy. */
  readonly blockedOperations?: readonly string[];
  /**
   * Whether a history this session creates joins the histories it may read and write.
   * Default true: an agent that makes a scratch history can then use it.
   */
  readonly includeCreated?: boolean;
  /** Whether the session may create histories at all. Default true. */
  readonly allowCreate?: boolean;
  /**
   * Which histories may be deleted: only the ones this session created (the default), or any
   * history it may write. Deleting is tighter than writing because it cannot be taken back by
   * the agent that did it by mistake.
   */
  readonly destructive?: "created" | "writable";
}

/** What `get_server_info` tells a scoped session about its own limits. */
export interface ScopeSummary {
  readable_history_ids: string[];
  writable_history_ids: string[];
  created_history_ids: string[];
  create_allowed: boolean;
  /** Which histories may be deleted: "created" (by this session) or "writable". */
  destructive: "created" | "writable";
  blocked_operations: string[];
  note: string;
}

/** Whether a check is about reading a history or changing it. */
export type ScopeAccess = "read" | "write";

/**
 * How many ids a run's handle memory holds before the least recently seen are dropped.
 *
 * A run that lists a few thousand datasets and jobs stays well under it; one that walks a
 * whole server forgets its oldest ids, which then pass the kind check as unknown ids do.
 */
export const DEFAULT_HANDLE_CAPACITY = 10_000;

/**
 * Which ids this run has seen, and as what.
 *
 * A Map from the lowercased id to the kinds it was returned as, in least-recently-seen
 * order: `recordHandles` refreshes an id it has seen before and drops the oldest ones past
 * `capacity`. It is a Map and not a richer type so a host that wants to share one across
 * contexts, or seed it, can hold it as one.
 */
export class HandleRegistry extends Map<string, Set<HandleKind>> {
  readonly capacity: number;

  constructor(capacity: number = DEFAULT_HANDLE_CAPACITY) {
    super();
    if (!Number.isInteger(capacity) || capacity < 1) {
      throw new RangeError(`a handle registry needs a capacity of at least 1 (got ${capacity})`);
    }
    this.capacity = capacity;
  }
}

const capacityOf = (handles: Map<string, Set<HandleKind>>): number =>
  handles instanceof HandleRegistry ? handles.capacity : DEFAULT_HANDLE_CAPACITY;

/**
 * Remember that these ids were returned as `kind`. A no-op on a context with no handle memory.
 *
 * Anything that is not a non-empty string is skipped, so a call site can hand over whatever a
 * reply carried -- `jobs.map((j) => j.id)` -- without checking each entry first.
 */
export function recordHandles(
  ctx: Pick<GalaxyContext, "handles">,
  kind: HandleKind,
  ids: Iterable<unknown>,
): void {
  const handles = ctx.handles;
  if (!handles) return;
  const capacity = capacityOf(handles);
  for (const raw of ids) {
    if (typeof raw !== "string" || raw === "") continue;
    const id = raw.toLowerCase();
    const seen = handles.get(id);
    if (seen) {
      // Refresh recency: a Map iterates in insertion order, so re-inserting moves it last.
      handles.delete(id);
      seen.add(kind);
      handles.set(id, seen);
      continue;
    }
    while (handles.size >= capacity) {
      const oldest = handles.keys().next();
      if (oldest.done) break;
      handles.delete(oldest.value);
    }
    handles.set(id, new Set([kind]));
  }
}

/** The kind a Galaxy `src` names, for the `{src, id}` references replies carry. */
const SRC_KINDS: Readonly<Record<string, HandleKind>> = {
  hda: "hda",
  hdca: "hdca",
  dce: "dataset_collection_element",
  job: "job",
};

/**
 * Record the `{src, id}` references a reply carries -- a job's inputs and outputs, an
 * invocation's inputs, a tool request's jobs. Entries that are not such a reference are skipped.
 */
export function recordRefs(ctx: Pick<GalaxyContext, "handles">, refs: Iterable<unknown>): void {
  if (!ctx.handles) return;
  for (const ref of refs) {
    if (!ref || typeof ref !== "object") continue;
    const { src, id } = ref as { src?: unknown; id?: unknown };
    const kind = typeof src === "string" ? SRC_KINDS[src] : undefined;
    if (kind) recordHandles(ctx, kind, [id]);
  }
}

/**
 * Record what a tool submission answered with -- POST /api/tools, /api/tools/fetch and the
 * user-tool run all answer in the one shape: the jobs queued, the datasets they will write and
 * the collections they will fill. Fields that are not lists are skipped.
 */
export function recordUploadResult(
  ctx: Pick<GalaxyContext, "handles">,
  result: {
    jobs?: unknown;
    outputs?: unknown;
    output_collections?: unknown;
    implicit_collections?: unknown;
  },
): void {
  if (!ctx.handles) return;
  const ids = (list: unknown): unknown[] =>
    Array.isArray(list) ? list.map((item) => (item as { id?: unknown } | null)?.id) : [];
  recordHandles(ctx, "job", ids(result.jobs));
  recordHandles(ctx, "hda", ids(result.outputs));
  recordHandles(ctx, "hdca", ids(result.output_collections));
  recordHandles(ctx, "hdca", ids(result.implicit_collections));
}

/** The kinds this run has seen an id as, or undefined for an id it has never seen. */
export function handleKindsOf(
  ctx: Pick<GalaxyContext, "handles">,
  id: string,
): ReadonlySet<HandleKind> | undefined {
  const seen = ctx.handles?.get(id.toLowerCase());
  return seen && seen.size > 0 ? seen : undefined;
}

/** The `history_id` a record carries, when it carries one. */
export function historyIdOf(record: unknown): string | undefined {
  if (!record || typeof record !== "object" || Array.isArray(record)) return undefined;
  const id = (record as { history_id?: unknown }).history_id;
  return typeof id === "string" && id !== "" ? id : undefined;
}

const sameId = (a: string, b: string): boolean => a.toLowerCase() === b.toLowerCase();

/** The histories this session created that the scope counts, or none when it does not. */
export function createdHistories(ctx: Pick<GalaxyContext, "scope" | "createdHistories">): readonly string[] {
  return ctx.scope?.includeCreated === false ? [] : [...(ctx.createdHistories ?? [])];
}

/** Remember a history this session created. A no-op on a context that keeps no such set. */
export function recordCreatedHistory(ctx: Pick<GalaxyContext, "createdHistories">, id: unknown): void {
  if (typeof id === "string" && id !== "") ctx.createdHistories?.add(id.toLowerCase());
}

/**
 * The histories a scope allows for this access: the host's list first, then the ones the
 * session created, then (for a read) the read-only ones.
 */
export function allowedHistories(
  scope: OperationScope,
  access: ScopeAccess,
  created: readonly string[] = [],
): readonly string[] {
  const writable = [...scope.historyIds, ...created.filter((c) => !scope.historyIds.some((h) => sameId(h, c)))];
  return access === "write" ? writable : [...writable, ...(scope.readOnlyHistoryIds ?? [])];
}

/** Whether `historyId` may be read, or changed, under this scope. */
export function historyAllowed(
  scope: OperationScope,
  historyId: string,
  access: ScopeAccess,
  created: readonly string[] = [],
): boolean {
  return allowedHistories(scope, access, created).some((allowed) => sameId(allowed, historyId));
}

/**
 * The histories a scoped session may see, of the ones Galaxy listed.
 *
 * The one narrowing an op applies itself rather than leaving to the registry: the history
 * listing is fetched whole and paged here, and a page and its total only describe the same
 * set when the rows outside the scope are dropped before the paging. The decision is still
 * this module's; unscoped, the rows come back as they were.
 */
export function visibleHistories<T>(ctx: Pick<GalaxyContext, "scope" | "createdHistories">, rows: readonly T[]): T[] {
  const { scope } = ctx;
  if (!scope) return [...rows];
  const created = createdHistories(ctx);
  return rows.filter((row) => {
    const id = (row as { id?: unknown } | null)?.id;
    return typeof id === "string" && historyAllowed(scope, id, "read", created);
  });
}

/** Whether the session may create a history: not switched off, and create_history not blocked. */
export function createAllowed(scope: OperationScope): boolean {
  return scope.allowCreate !== false && !(scope.blockedOperations ?? []).includes("create_history");
}

/** Whether `historyId` may be deleted: created here, or any writable one when the host said so. */
export function historyDeletable(scope: OperationScope, historyId: string, created: readonly string[]): boolean {
  if (scope.destructive === "writable") return historyAllowed(scope, historyId, "write", created);
  return created.some((c) => sameId(c, historyId));
}

/** The scope as a session reads it from get_server_info. */
export function scopeSummary(ctx: Pick<GalaxyContext, "scope" | "createdHistories">): ScopeSummary | undefined {
  const { scope } = ctx;
  if (!scope) return undefined;
  const created = createdHistories(ctx);
  const writable = [...allowedHistories(scope, "write", created)];
  const readable = [...allowedHistories(scope, "read", created)];
  const create = createAllowed(scope);
  const destructive = scope.destructive ?? "created";
  const note =
    `This session is scoped. It may read histories: ${list(readable)}; write to: ${list(writable)}. ` +
    (create
      ? scope.includeCreated === false
        ? "It may create histories but not then use them. "
        : "It may create histories, and use the ones it creates. "
      : "It may not create histories. ") +
    (destructive === "writable"
      ? "It may delete any history it may write."
      : "It may delete only histories it created itself.");
  return {
    readable_history_ids: readable,
    writable_history_ids: writable,
    created_history_ids: [...created],
    create_allowed: create,
    destructive,
    blocked_operations: [...(scope.blockedOperations ?? [])],
    note,
  };
}

/**
 * The op's own parameter name, spelled the way the wire spells it.
 *
 * The package's refusals name parameters in the Python server's spelling (`dataset_id`,
 * `history_id`), because that is what an MCP client sent and what the sentence it would have
 * read from that server says; the CLI's flags are the same words with dashes.
 */
export function wireName(key: string): string {
  return key.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase();
}

const list = (ids: readonly string[]): string => (ids.length === 0 ? "none" : ids.join(", "));

/** "Histories this run may write: a, b; read-only: c" -- ids only, never names. */
function allowanceSentence(scope: OperationScope, access: ScopeAccess, created: readonly string[] = []): string {
  if (access === "read") return `Histories this run may read: ${list(allowedHistories(scope, "read", created))}`;
  const readOnly = scope.readOnlyHistoryIds ?? [];
  return (
    `Histories this run may write: ${list(allowedHistories(scope, "write", created))}` +
    (readOnly.length > 0 ? `; read-only: ${list(readOnly)}` : "")
  );
}

export function blockedOperationSentence(opName: string): string {
  return `${opName} is not available in this run: the host's scope blocks it. Nothing was sent to Galaxy.`;
}

export function historyOutOfScopeSentence(
  opName: string,
  param: string,
  historyId: string,
  scope: OperationScope,
  access: ScopeAccess,
  created: readonly string[] = [],
): string {
  const readOnlyTarget = access === "write" && historyAllowed(scope, historyId, "read", created);
  const why = readOnlyTarget
    ? `${wireName(param)} '${historyId}' is read-only in this run and ${opName} changes it`
    : `${wireName(param)} '${historyId}' is outside this run's scope`;
  return `${why}. ${allowanceSentence(scope, access, created)}. Nothing was sent to Galaxy.`;
}

/** `params` is one name, or the alternatives any one of which will do. */
export function missingScopedParamSentence(
  opName: string,
  params: string | readonly string[],
  scope: OperationScope,
  created: readonly string[] = [],
): string {
  const names = (typeof params === "string" ? [params] : params).map(wireName).join(" or ");
  return (
    `${opName} needs ${names} in this run: without it Galaxy would use its current ` +
    `history, which is not known to be one this run may write. ` +
    `${allowanceSentence(scope, "write", created)}. Nothing was sent to Galaxy.`
  );
}

export function createNotAllowedSentence(opName: string): string {
  return `${opName} would create a history, and this run may not create histories. Nothing was sent to Galaxy.`;
}

export function notDeletableSentence(
  opName: string,
  param: string,
  historyId: string,
  scope: OperationScope,
  created: readonly string[],
): string {
  const rule =
    scope.destructive === "writable"
      ? `Histories this run may delete: ${list(allowedHistories(scope, "write", created))}`
      : `This run may delete only histories it created itself: ${list(created)}`;
  return (
    `${opName} would delete ${wireName(param)} '${historyId}', which this run may not delete. ` +
    `${rule}. Nothing was sent to Galaxy.`
  );
}

export function ownerOutOfScopeSentence(
  opName: string,
  noun: string,
  id: string,
  owner: string,
  scope: OperationScope,
  access: ScopeAccess,
  when: "before" | "after",
  created: readonly string[] = [],
): string {
  const readOnlyOwner = access === "write" && historyAllowed(scope, owner, "read", created);
  const why = readOnlyOwner
    ? `belongs to history '${owner}', which is read-only in this run, and ${opName} changes it`
    : `belongs to history '${owner}', which is outside this run's scope`;
  const outcome = when === "before" ? `${opName} was not run.` : "It was not returned.";
  return `${noun} '${id}' ${why}. ${allowanceSentence(scope, access, created)}. ${outcome}`;
}

export function kindMismatchSentence(
  param: string,
  id: string,
  seen: ReadonlySet<HandleKind>,
  expected: HandleKind,
): string {
  const nouns = [...seen].map((k) => `a ${HANDLE_NOUNS[k]}`);
  const as = nouns.length === 1 ? nouns[0]! : `${nouns.slice(0, -1).join(", ")} and ${nouns[nouns.length - 1]}`;
  return (
    `${wireName(param)} '${id}' was returned by an earlier call as ${as}; ` +
    `${wireName(param)} expects a ${HANDLE_NOUNS[expected]}. Nothing was sent to Galaxy.`
  );
}
