import type { ZodRawShape } from "zod";
import type { GalaxyContext } from "../context";
import { GalaxyError, GalaxyVersionError } from "../errors";
import { parseRequirement, requirementSentence, satisfiesRequirement } from "../version";
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

/** Wrap an op for a surface: catch typed errors, apply project() metadata. */
export async function runWithEnvelope<Shape extends ZodRawShape, O>(
  op: Operation<Shape, O>,
  input: InputOf<Shape>,
  ctx: GalaxyContext,
): Promise<GalaxyResult<O>> {
  try {
    const data = await runOperation(op, input, ctx);
    const meta = op.project?.(data, input) ?? {};
    return { data, success: true, ...meta };
  } catch (err) {
    if (err instanceof GalaxyError) {
      return { data: undefined as unknown as O, success: false, message: err.message, errorKind: err.kind };
    }
    throw err; // non-Galaxy errors are bugs -- let them surface
  }
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
