/**
 * The order a workflow's steps are walked in.
 *
 * A workflow's `steps` is a JSON object, and the two languages disagree about what
 * an object's order is: Python keeps the order the keys arrived in, JavaScript
 * visits integer-like keys in ascending numeric order first and the rest in arrival
 * order. Neither can cheaply pretend to be the other, so both servers now sort
 * explicitly, by this rule: steps are visited by numeric key ascending for keys
 * that are non-negative integers written canonically -- no leading zeros, no sign
 * -- and then the remaining keys in the order they arrived. The same rule says
 * which key is a step index.
 *
 * Faithful port of Python `steps_in_order` / `canonical_step_index`.
 */

const CANONICAL_STEP_KEY = /^(?:0|[1-9][0-9]*)$/;

/**
 * The step index a key names, or null when the key is not one.
 *
 * A canonical non-negative integer and nothing else: not `007`, not `+1`, not
 * `1e2`, not a hash. Those are keys a workflow should not have, and the two
 * languages read them differently enough that guessing costs more than skipping.
 */
export function canonicalStepIndex(key: string): number | null {
  return CANONICAL_STEP_KEY.test(key) ? Number(key) : null;
}

/**
 * A workflow's steps in the one order both servers agree on.
 *
 * The numeric keys are compared as decimal strings -- shorter first, then
 * lexicographically -- which is their numeric order exactly, for a key of any
 * length, and does not go through a float. Nothing here leans on the order
 * `Object.entries` happens to return: every key the runtime would have hoisted is
 * a canonical one, and those are all re-sorted here.
 */
export function stepsInOrder(steps: Record<string, unknown>): Array<[string, unknown]> {
  const numbered: Array<[string, unknown]> = [];
  const rest: Array<[string, unknown]> = [];
  for (const entry of Object.entries(steps)) {
    (CANONICAL_STEP_KEY.test(entry[0]) ? numbered : rest).push(entry);
  }
  numbered.sort(
    ([a], [b]) => a.length - b.length || (a < b ? -1 : a > b ? 1 : 0),
  );
  return [...numbered, ...rest];
}
