import type { ZodRawShape } from "zod";
import type { EnvelopeFacts, GalaxyContext } from "../context";
import { GalaxyError, GalaxyHandleKindError, GalaxyScopeError, GalaxyVersionError, redirectRefusal } from "../errors";
import { pyFormatError, pyLibraryText } from "../python-failure";
import {
  HANDLE_NOUNS,
  blockedOperationSentence,
  createAllowed,
  createNotAllowedSentence,
  createdHistories,
  historyDeletable,
  notDeletableSentence,
  scopeSummary,
  handleKindsOf,
  historyAllowed,
  historyOutOfScopeSentence,
  kindMismatchSentence,
  missingScopedParamSentence,
  ownerOutOfScopeSentence,
  type HandleKind,
  type OperationScope,
  type ScopeAccess,
} from "../scope";
import { parseRequirement, requirementSentence, satisfiesRequirement } from "../version";
import { trimToBudget } from "./pagination";
import type { AnyOperation, GalaxyResult, InputOf, Operation, ScopeSpec } from "./types";

/** Something that can carry a Galaxy requirement: an op, or a copy of one. */
type Claim = { name: string; requires?: { galaxy: string } };

/**
 * Check every requirement that applies against ONE reading of the version, and hand back a
 * context carrying that reading.
 *
 * The pin is the whole point. Several guards can run inside a single call -- this one, the one
 * on the op's own run, another on an op it composes -- and they must not be able to reach
 * different conclusions. Reading the version once and passing it down makes that true by
 * construction, rather than by hoping two lookups land close enough together in time.
 *
 * Every claim is checked, not the first or the strictest: a copy of an op is held to its own
 * requirement AND to the registered op's, so it can tighten what it needs but never loosen it.
 * Nothing is looked up when no claim declares a requirement, so the version request only
 * happens where it could change the answer, and an unknown version passes -- the flag is here
 * to spare a caller a round trip that cannot succeed, not to gate a server we failed to read.
 */
export async function guardVersion(
  ctx: GalaxyContext,
  ...claims: (Claim | undefined)[]
): Promise<GalaxyContext> {
  const required = claims.filter((c): c is Claim & { requires: { galaxy: string } } =>
    Boolean(c?.requires),
  );
  if (required.length === 0) return ctx;

  const report = await ctx.galaxyVersion?.();
  const version = report?.version;
  if (version) {
    for (const claim of required) {
      if (satisfiesRequirement(version, claim.requires.galaxy)) continue;
      const want = parseRequirement(claim.requires.galaxy);
      // server.py, _assert_version_supported: the last clause is the useful half of it --
      // an agent that reads "needs 26.1" still has to be told the call did not happen.
      throw new GalaxyVersionError(
        `${claim.name} needs Galaxy ${want.major}.${want.minor} or newer; ` +
          `this server reports ${version.raw}. Nothing was sent to Galaxy.`,
      );
    }
  }
  return report ? { ...ctx, galaxyVersion: () => Promise.resolve(report) } : ctx;
}

/** Refuse an op the connected Galaxy is too old for, before a single request goes out. */
export async function assertVersionSupported(op: Claim, ctx: GalaxyContext): Promise<void> {
  await guardVersion(ctx, op);
}

// ---------------------------------------------------------------------------
// The scope guard: everything a scoped context refuses or narrows, in this one place.
// ---------------------------------------------------------------------------

/**
 * The parameter names that always carry a Galaxy id of a known kind.
 *
 * Not the policy -- that is each op's own `ids` declaration -- but the lint behind it: an op
 * that takes a parameter spelled like one of these and does not classify it fails to register,
 * so a new op cannot slip past the scope guard by forgetting. A parameter under any other name
 * (`pageId`, `toolUuid`, `trsId`) is not a history-scoped resource and is not an id of a kind
 * the run remembers, so it needs no entry.
 */
export const KNOWN_ID_PARAMS: Readonly<Record<string, HandleKind>> = Object.freeze({
  historyId: "history",
  datasetId: "hda",
  collectionId: "hdca",
  jobId: "job",
  invocationId: "invocation",
  workflowId: "workflow",
  toolRequestId: "tool_request",
});

/** What the registry kept of an op's scope declarations, out of reach of a later edit. */
interface ScopeClaims {
  readonly name: string;
  readonly ids: Readonly<Record<string, HandleKind>>;
  readonly access: ScopeAccess;
  readonly spec: ScopeSpec<unknown, unknown>;
}

/** The id strings an argument holds: one, a list of them, or none. Empty strings are no id. */
function idValues(value: unknown): string[] {
  if (typeof value === "string") return value === "" ? [] : [value];
  if (Array.isArray(value)) return value.filter((v): v is string => typeof v === "string" && v !== "");
  return [];
}

/** How the op's record is named in a refusal: "dataset 'd1' belongs to history ...". */
function recordNoun(claims: ScopeClaims, input: Record<string, unknown>): { noun: string; id: string } {
  for (const [param, kind] of Object.entries(claims.ids)) {
    if (kind === "history") continue;
    const [id] = idValues(input[param]);
    if (id !== undefined) return { noun: HANDLE_NOUNS[kind].replace(/ id$/, ""), id };
  }
  return { noun: "record", id: "" };
}

/**
 * Everything checked before the op runs that needs no request: the blocked list, then every id
 * argument -- its kind against what the run has seen the id as, and a history against the scope
 * -- then the parameters a scoped run must supply.
 *
 * The kind check runs whenever the context remembers handles, scope or no scope, because a
 * host can ask for it alone. Everything else needs a scope. An id the run has never seen
 * passes the kind check -- Galaxy's ids carry no type, so there is nothing to compare.
 */
function guardScopeArguments(claims: ScopeClaims, input: Record<string, unknown>, ctx: GalaxyContext): void {
  const { scope } = ctx;
  if (scope?.blockedOperations?.includes(claims.name)) {
    throw new GalaxyScopeError(blockedOperationSentence(claims.name));
  }
  const created = createdHistories(ctx);
  const destroys = scope !== undefined && claims.spec.destroys?.(input) === true;
  for (const [param, kind] of Object.entries(claims.ids)) {
    for (const id of idValues(input[param])) {
      const seen = handleKindsOf(ctx, id);
      if (seen && !seen.has(kind)) {
        throw new GalaxyHandleKindError(kindMismatchSentence(param, id, seen, kind));
      }
      if (!scope || kind !== "history") continue;
      if (!historyAllowed(scope, id, claims.access, created)) {
        throw new GalaxyScopeError(
          historyOutOfScopeSentence(claims.name, param, id, scope, claims.access, created),
        );
      }
      // Deleting is held to a tighter rule than writing: a writable history the session did
      // not create is still not its to delete, unless the host said so.
      if (destroys && !historyDeletable(scope, id, created)) {
        throw new GalaxyScopeError(notDeletableSentence(claims.name, param, id, scope, created));
      }
    }
  }
  if (!scope) return;
  for (const entry of claims.spec.requires ?? []) {
    const names = typeof entry === "string" ? [entry] : entry;
    if (!names.some((param) => idValues(input[param]).length > 0)) {
      throw new GalaxyScopeError(missingScopedParamSentence(claims.name, entry, scope, created));
    }
  }
  if (claims.spec.creates?.(input) && !createAllowed(scope)) {
    throw new GalaxyScopeError(createNotAllowedSentence(claims.name));
  }
}

/**
 * The one pre-call check that costs a request: for an op that declared `ownerOf`, ask which
 * history its target belongs to and refuse before the op's own request -- a cancel, a write to
 * disk -- when it is outside the scope. Only under a scope; nothing is asked otherwise.
 */
async function guardScopeOwner(claims: ScopeClaims, input: Record<string, unknown>, ctx: GalaxyContext): Promise<void> {
  const { scope } = ctx;
  if (!scope || !claims.spec.ownerOf) return;
  let owner: string | null | undefined;
  try {
    owner = await claims.spec.ownerOf(input, ctx);
  } catch {
    // The lookup could not say. The op runs and stands or falls on its own request, which
    // for a missing or unreadable record is the same failure, worded by the op.
    owner = undefined;
  }
  const created = createdHistories(ctx);
  if (owner && !historyAllowed(scope, owner, claims.access, created)) {
    const { noun, id } = recordNoun(claims, input);
    throw new GalaxyScopeError(
      ownerOutOfScopeSentence(claims.name, noun, id, owner, scope, claims.access, "before", created),
    );
  }
}

/**
 * What a scoped run may see of the op's output: a record is refused when it belongs to a
 * history outside the scope, a listing is narrowed to the scoped histories, and an account
 * record is cut down. Without a scope the output goes out untouched.
 */
function narrowOutput(
  claims: ScopeClaims,
  output: unknown,
  input: Record<string, unknown>,
  ctx: GalaxyContext,
): unknown {
  const scope: OperationScope | undefined = ctx.scope;
  if (!scope) return output;
  // Read after the op ran, so a history the op itself just created counts.
  const created = createdHistories(ctx);
  const owner = claims.spec.owner?.(output);
  if (owner && !historyAllowed(scope, owner, claims.access, created)) {
    const { noun, id } = recordNoun(claims, input);
    throw new GalaxyScopeError(
      ownerOutOfScopeSentence(claims.name, noun, id, owner, scope, claims.access, "after", created),
    );
  }
  let narrowed = output;
  if (claims.spec.filter) {
    narrowed = claims.spec.filter(narrowed, (historyId) =>
      typeof historyId === "string" && historyAllowed(scope, historyId, "read", created),
    );
  }
  if (claims.spec.restrict) narrowed = claims.spec.restrict(narrowed);
  const summary = claims.spec.annotate ? scopeSummary(ctx) : undefined;
  if (claims.spec.annotate && summary) narrowed = claims.spec.annotate(narrowed, summary);
  return narrowed;
}

/**
 * Read an op's scope declarations once, at registration, and check them against its schema.
 *
 * Kept as plain copies in the registry's closure, like the version requirement: the op is
 * frozen afterwards, but a copy of it could otherwise carry different declarations to the
 * same registered run.
 */
function scopeClaims(op: AnyOperation): ScopeClaims {
  const declared: Record<string, HandleKind> = { ...(op.ids ?? {}) } as Record<string, HandleKind>;
  for (const param of Object.keys(declared)) {
    if (!(param in op.input)) {
      throw new Error(`operation "${op.name}" declares an id parameter "${param}" its input does not have`);
    }
    if (!(declared[param]! in HANDLE_NOUNS)) {
      throw new Error(`operation "${op.name}" declares "${param}" as an unknown id kind "${declared[param]}"`);
    }
  }
  for (const param of Object.keys(op.input)) {
    if (param in KNOWN_ID_PARAMS && !(param in declared)) {
      throw new Error(
        `operation "${op.name}" takes "${param}" but declares no \`ids\` entry for it, ` +
          "so the scope guard cannot classify it",
      );
    }
  }
  const spec = { ...(op.scope ?? {}) } as ScopeSpec<unknown, unknown>;
  for (const param of (spec.requires ?? []).flat()) {
    if (!(param in op.input)) {
      throw new Error(`operation "${op.name}" requires a parameter "${param}" its input does not have`);
    }
  }
  if (op.ids) Object.freeze(op.ids);
  if (op.scope) Object.freeze(op.scope);
  return {
    name: op.name,
    ids: Object.freeze(declared),
    access: spec.access ?? (op.readOnly === false ? "write" : "read"),
    spec: Object.freeze(spec),
  };
}

/**
 * Run an op, guarded by the requirement of the op it was handed.
 *
 * The context handed on is the pinned one, so the guard on the op's own run -- and any op this
 * one composes -- works from the same reading of the version this check used. Checking again
 * further in is then free and cannot contradict.
 */
export async function runOperation<Shape extends ZodRawShape, O>(
  op: Operation<Shape, O>,
  input: InputOf<Shape>,
  ctx: GalaxyContext,
): Promise<O> {
  const pinned = await guardVersion(ctx, op);
  return op.run(input, pinned);
}

/** The line a surface shows for an op: its summary, and what it needs from the server. */
export function describeOperation(op: { summary: string; requires?: { galaxy: string } }): string {
  return op.requires ? `${op.summary} ${requirementSentence(op.requires.galaxy)}` : op.summary;
}

/**
 * Help text with the op's own parameter names spelled the way the surface asking spells them.
 *
 * One sentence has to be true in three dialects: the MCP wire takes `section_id`, the command
 * line takes `--section-id`, and an op called from TypeScript takes `sectionId`. Descriptions
 * are written in the op's own key and each surface respells them on the way out, so a tool
 * that tells a caller what to pass names something that caller can actually pass.
 *
 * Only the camelCase keys are respelled. A single-word key reads the same on every surface,
 * and half of them -- `name`, `inputs`, `version`, `content`, `path` -- are ordinary English
 * in these sentences, where substituting them would wreck the prose rather than fix it.
 */
export function spellParamNames(
  text: string,
  input: ZodRawShape,
  spell: (key: string) => string,
): string {
  // Longest first, so a key that is a prefix of another cannot claim the shorter match.
  // The keys are TypeScript identifiers, which is why they go into the pattern unescaped.
  const keys = Object.keys(input)
    .filter((key) => /[A-Z]/.test(key))
    .sort((a, b) => b.length - a.length);
  if (keys.length === 0) return text;
  return text.replace(new RegExp(`\\b(?:${keys.join("|")})\\b`, "g"), (key) => spell(key));
}

/**
 * Wrap an op for a surface: catch typed errors, project it into the envelope, fit
 * the budget.
 *
 * The envelope is the Python server's, key for key. `data` is whatever the op's
 * projection says it is -- for a paged op the bare page, with the window lifted
 * beside it rather than travelling twice -- and `count` and `pagination` are
 * always sent, null where the tool has none, because that is what a pydantic
 * model puts on the wire.
 *
 * The result type is `unknown` rather than the op's own: a projection is free to
 * emit something other than what `run` returned, and it usually does.
 */
export async function runWithEnvelope<Shape extends ZodRawShape, O>(
  op: Operation<Shape, O>,
  input: InputOf<Shape>,
  ctx: GalaxyContext,
  /**
   * How this surface will serialise the envelope, so the budget is measured on the
   * bytes it actually emits. The default is the compact single line the MCP text
   * block carries; the CLI prints indented JSON and passes that instead, because a
   * budget measured against a shorter rendering than the one printed is not a
   * budget.
   */
  serialize: (result: GalaxyResult<unknown>) => string = (result) => JSON.stringify(result),
): Promise<GalaxyResult<unknown>> {
  try {
    // One collector, belonging to this call and passed down with the context, for
    // the facts an op learns while it runs that its return value has no room for
    // -- a total off a response header, a count from a second request. Keying
    // them on the object run() returned is the same idea and is wrong: a client
    // that answers two calls with one frozen array makes the two calls
    // indistinguishable, and the second one's numbers won.
    const facts: EnvelopeFacts = new Map();
    const data = await runOperation(op, input, { ...ctx, envelopeFacts: facts });
    const envelope = (d: O): GalaxyResult<unknown> => {
      const projected = op.project?.(d, input, facts) ?? {};
      return {
        data: "data" in projected ? projected.data : d,
        success: true,
        ...(projected.message === undefined ? {} : { message: projected.message }),
        count: projected.count ?? null,
        pagination: projected.pagination ?? null,
      };
    };
    // The budget is measured on the projected envelope rather than on the page,
    // because the projection is what gets serialised -- a page that fits before
    // the window is lifted out of it is not a page that fits. Trimming
    // re-projects, so a cut page's message, count and pagination are its own.
    return envelope(op.budget ? trimToBudget(data, op.budget, (d) => serialize(envelope(d))) : data);
  } catch (err) {
    if (err instanceof GalaxyError) {
      return {
        data: undefined as unknown as O,
        success: false,
        message: pythonFailureSentence(op, err, input),
        errorKind: err.kind,
      };
    }
    throw err; // non-Galaxy errors are bugs -- let them surface
  }
}

/**
 * What the Python server would have said about this failure.
 *
 * Its tools catch whatever their client raised and word one sentence out of three things: an
 * action, the exception's own text, and a context dict. Only the first and third belong to
 * the tool, so they live on the op (see FailureContract) and the middle one is rebuilt here
 * from the reply the request left behind.
 *
 * A failure with no reply behind it is a refusal of ours, and its message is already the
 * whole sentence -- the same reason the other server re-raises its own ValueErrors
 * untouched. An op that declares no contract keeps its own short message, which is the
 * library sentence a caller reading the result would have seen before this existed.
 */
export function pythonFailureSentence<Shape extends ZodRawShape, O>(
  op: Operation<Shape, O>,
  err: GalaxyError,
  input: InputOf<Shape>,
): string {
  const facts = err.http;
  const contract = op.failure;
  if (!facts || !contract) return err.message;
  const shape = typeof contract.shape === "function" ? contract.shape(facts) : contract.shape;
  // An unfollowed 3xx is ours to word: Python's text for one quotes the body and never the
  // Location, and its raise-for-status ops do not fail on a 3xx at all (see python-failure.ts).
  const text =
    facts.redirectedTo !== undefined
      ? redirectRefusal(facts.status, facts.redirectedTo || undefined)
      : pyLibraryText(shape, facts);
  const own = contract.sentence?.(text, facts.status, input);
  if (own !== undefined) return own;
  if (contract.action === undefined) return text;
  return pyFormatError(contract.action, text, facts.status, contract.context?.(input) ?? {});
}

/** The v1 registry. Populated as ops land (Tasks 8, 12, 15). */
export const allOperations: AnyOperation[] = [];

const registeredNames = new Set<string>();

export function register(op: AnyOperation): AnyOperation {
  // A requirement is read at import time so a typo fails the build rather than one call.
  if (op.requires) parseRequirement(op.requires.galaxy);

  // Registering twice would wrap the wrapper, so one call would consult the version once per
  // registration. Caught here rather than left to surprise someone later.
  if (registeredNames.has(op.name)) {
    throw new Error(`operation "${op.name}" is already registered`);
  }
  if (allOperations.includes(op)) {
    throw new Error(`this operation object is already registered (as "${op.name}")`);
  }

  // What the op asked for when it registered, kept as a string rather than as a handle on the
  // object it came from. A shallow copy shares that nested object, so `{ ...op }.requires.galaxy
  // = ">=26.0"` would otherwise rewrite what the original enforces, for every caller. A string
  // in a closure cannot be reached, let alone edited. Freezing the object as well turns that
  // assignment into a TypeError instead of a silent no-op -- belt for the braces below.
  const registeredName = op.name;
  const registeredSpec = op.requires?.galaxy;
  if (op.requires) Object.freeze(op.requires);
  const claims = scopeClaims(op);

  // Replace run on the object itself, which is what makes the guard unreachable-around rather
  // than merely available: the module's named export, the entry in allOperations and anything a
  // caller destructures are all this one object, so a raw run left on it is a way straight past
  // the check. The original survives only in this closure and is deliberately not exported --
  // there is no unguarded entry point to reach for by accident.
  const unguarded = op.run.bind(op);
  op.run = async function (this: Claim | undefined, input: never, ctx: GalaxyContext) {
    // The requirement registered here always applies. A copy this was called on is checked as
    // well when it carries a different one, so a copy can ask for MORE than the op it was made
    // from but never for less -- whichever direction it differs in, and whether it arrived as a
    // receiver or, with no receiver at all, as a destructured function.
    //
    // Only the NAME follows the receiver, so a refusal blames whatever the caller called it --
    // the same name the surfaces use. The spec comes from the closure and never from an object.
    const registeredClaim = registeredSpec
      ? { name: this?.name ?? registeredName, requires: { galaxy: registeredSpec } }
      : undefined;
    const receiver =
      this?.requires && this.requires.galaxy !== registeredSpec ? this : undefined;
    // The scope's argument checks first, because they cost no request and a refused call must
    // not probe the server's version either; then the version; then the owner lookup, which is
    // a request; then the op, whose output the scope narrows on the way out.
    const args = (input ?? {}) as Record<string, unknown>;
    const scoped = { ...claims, name: this?.name ?? registeredName };
    guardScopeArguments(scoped, args, ctx);
    const pinned = await guardVersion(ctx, registeredClaim, receiver);
    await guardScopeOwner(scoped, args, pinned);
    return narrowOutput(scoped, await unguarded(input, pinned), args, ctx);
  };

  registeredNames.add(op.name);
  allOperations.push(op);

  // Sealed once it is wired up. The wrapper above is only a guard while it is the function that
  // actually runs, and until now `op.run = somethingElse` quietly removed it, leaving a direct
  // call less guarded than every other way in. Freezing also stops `requires`, `name` and the
  // rest being swapped afterwards, which is what let an op advertise one version and enforce
  // another. Assignment throws in strict mode rather than passing unnoticed.
  return Object.freeze(op);
}
