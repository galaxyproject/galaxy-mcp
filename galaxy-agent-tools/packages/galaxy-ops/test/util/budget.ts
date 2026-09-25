import { expect } from "vitest";
import { OUTPUT_BUDGET_BYTES } from "../../src/operations/pagination";
import { runWithEnvelope } from "../../src/operations/registry";
import { mcpBytesOf } from "./byte-budget";
import type { GalaxyContext } from "../../src/context";
import type { AnyOperation, GalaxyResult } from "../../src/operations/types";

/** The bytes a client is handed for one call: the envelope, as the surface serialises it. */
export const wireBytes = (result: unknown): number =>
  new TextEncoder().encode(JSON.stringify(result)).length;

/**
 * Walk from offset 0 to the last page, checking every page on the way.
 *
 * The port of the Python suite's `walk_every_page`. Ceilings are limits on the
 * request, so what comes back at one is whatever fits; what is checked is that it
 * fits, that it says how to reach the rest, and that following that advice sees
 * every item exactly once.
 */
export async function walkEveryPage(opts: {
  op: AnyOperation;
  ctx: GalaxyContext;
  /** The op's input for one page. */
  input: (limit: number, offset: number) => Record<string, unknown>;
  /** The rows inside a result's data, whatever key this op puts them under. */
  rows: (data: unknown) => unknown[];
  /** A stable identity per row, so a duplicate is visible. */
  id: (row: unknown) => string;
  /** The ceiling to ask for, which is the number the Python tool caps this tool at. */
  limit: number;
  label: string;
}): Promise<string[]> {
  const call = (limit: number, offset: number) =>
    runWithEnvelope(opts.op as never, opts.input(limit, offset) as never, opts.ctx) as Promise<
      GalaxyResult<unknown>
    >;

  await assertTheFixtureIsOverBudget(opts, call);

  const seen: string[] = [];
  let offset = 0;
  for (let page = 0; page < 500; page += 1) {
    const result = await call(opts.limit, offset);
    const size = wireBytes(result);
    expect(size, `${opts.label} page at offset ${offset} reaches the client as ${size} bytes`).toBeLessThanOrEqual(
      OUTPUT_BUDGET_BYTES,
    );
    const rows = opts.rows(result.data);
    seen.push(...rows.map(opts.id));
    if (!result.pagination?.hasNext) {
      expect(new Set(seen).size, `${opts.label} returned an item twice`).toBe(seen.length);
      return seen;
    }
    expect(result.pagination.nextOffset, `${opts.label} advanced past what it returned`).toBe(
      offset + rows.length,
    );
    offset = result.pagination.nextOffset!;
  }
  throw new Error(`${opts.label} never reached its last page`);
}

/**
 * The corpus has to be one the budget really has to cut, or nothing is proved.
 *
 * An empty result fits every budget and so does a small one; either would let a
 * test that only checks "the page fits" pass while the cutting never ran. The
 * uncut size comes from the op's own run, which is the page before the surface
 * trims it.
 */
async function assertTheFixtureIsOverBudget(
  opts: { op: AnyOperation; ctx: GalaxyContext; input: (l: number, o: number) => Record<string, unknown>; rows: (d: unknown) => unknown[]; limit: number; label: string },
  call: (limit: number, offset: number) => Promise<GalaxyResult<unknown>>,
): Promise<void> {
  const input = opts.input(opts.limit, 0);
  const uncut = await opts.op.run(input as never, opts.ctx);
  expect(opts.rows(uncut).length, `${opts.label}: the fixture produced nothing to cut`).toBeGreaterThan(0);
  const size = mcpBytesOf(opts.op, uncut, input);
  expect(size, `${opts.label}: an uncut page is only ${size} bytes, so the budget is never reached`).toBeGreaterThan(
    OUTPUT_BUDGET_BYTES,
  );
  // And the surface really does cut it, rather than the walk below passing because
  // the op quietly returned fewer rows than it was asked for.
  const cut = await call(opts.limit, 0);
  expect(opts.rows(cut.data).length, `${opts.label}: the page was not cut`).toBeLessThan(
    opts.rows(uncut).length,
  );
  expect(cut.pagination?.trimmedForSize, `${opts.label}: a cut page must say it was cut`).toBe(true);
}
