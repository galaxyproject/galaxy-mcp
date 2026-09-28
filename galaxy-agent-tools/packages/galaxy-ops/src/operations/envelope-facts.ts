/**
 * Facts an op learns while it runs that the envelope needs and `run()` does not
 * return.
 *
 * There are two of them: how many pages matched, which Galaxy sends on a
 * `total_matches` response header rather than in the body, and how many items a
 * history holds, which costs a second request. Neither can travel in the op's
 * return value, because that value is the library's -- a `PageSummary[]`, a
 * history record -- and every direct caller already destructures it.
 *
 * So they travel beside the call instead. `runWithEnvelope` puts a fresh
 * collector on the context it passes down, `run` writes to it through
 * `recordFact`, and the projection reads it through `readFact`. One collector
 * per call is the whole point: keying on the object `run()` returned looked
 * equivalent and is not, because a client is free to answer two calls with the
 * same frozen array, and then the second call's total overwrote the first's.
 *
 * A caller that runs an op directly -- code mode, a TypeScript importer, an op
 * composing another -- passes a context with no collector on it, and every
 * `recordFact` is a no-op. Nothing about `run()` changes either way.
 */
import type { EnvelopeFacts, GalaxyContext } from "../context";

/**
 * One named fact and the type of its value.
 *
 * A symbol, so two ops cannot collide on a string, and the value type is carried
 * on a field that is never assigned: it exists to make `recordFact` and
 * `readFact` agree about what this fact holds.
 */
export interface EnvelopeFact<T> {
  readonly key: symbol;
  readonly value?: T;
}

/** Declare a fact. Module level, next to the op that records it. */
export const envelopeFact = <T>(name: string): EnvelopeFact<T> => ({ key: Symbol(name) });

/** Record a fact for this call's projection; a no-op when nobody is collecting. */
export function recordFact<T>(ctx: GalaxyContext, fact: EnvelopeFact<T>, value: T): void {
  ctx.envelopeFacts?.set(fact.key, value);
}

/** What this call's `run` recorded, if it recorded anything. */
export function readFact<T>(
  facts: EnvelopeFacts | undefined,
  fact: EnvelopeFact<T>,
): T | undefined {
  return facts?.get(fact.key) as T | undefined;
}
