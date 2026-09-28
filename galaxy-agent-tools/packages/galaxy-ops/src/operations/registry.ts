import type { ZodRawShape } from "zod";
import type { EnvelopeFacts, GalaxyContext } from "../context";
import { GalaxyError, GalaxyVersionError } from "../errors";
import { pyFormatError, pyLibraryText } from "../python-failure";
import { parseRequirement, requirementSentence, satisfiesRequirement } from "../version";
import { trimToBudget } from "./pagination";
import type { AnyOperation, GalaxyResult, InputOf, Operation } from "./types";

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
      throw new GalaxyVersionError(
        `${claim.name} needs Galaxy ${want.major}.${want.minor} or newer; this server reports ${version.raw}`,
      );
    }
  }
  return report ? { ...ctx, galaxyVersion: () => Promise.resolve(report) } : ctx;
}

/** Refuse an op the connected Galaxy is too old for, before a single request goes out. */
export async function assertVersionSupported(op: Claim, ctx: GalaxyContext): Promise<void> {
  await guardVersion(ctx, op);
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
  const text = pyLibraryText(shape, facts);
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
    const pinned = await guardVersion(ctx, registeredClaim, receiver);
    return unguarded(input, pinned);
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
