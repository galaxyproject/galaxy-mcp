import { describe, it, expect } from "vitest";
import { getHistoriesOp, getHistories } from "../../src/operations/get-histories";
import { mockClient } from "../util/mock-client";
import { mcpPayloadBytes } from "../util/byte-budget";
import { runWithEnvelope } from "../../src/operations/registry";
import { OUTPUT_BUDGET_BYTES } from "../../src/operations/pagination";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });
const serving = (rows: unknown[], seen?: (query: any) => void) =>
  mockClient({
    GET: (path, init) => {
      expect(path).toBe("/api/histories");
      seen?.(init.params.query);
      return { data: rows, response: { status: 200 } };
    },
  });
const named = (names: string[]) => names.map((name, k) => ({ id: `h${k + 1}`, name }));

describe("get_histories", () => {
  it("asks Galaxy for every history and windows the list here", async () => {
    let query: any;
    const out = await getHistories({ limit: 2 }, ctxWith(serving(named(["alpha", "beta", "gamma"]), (q) => (query = q))));
    // No window on the wire: the other surface ends up fetching everything as well,
    // because it asks a second time unpaged to count what matched.
    expect(query).toEqual({ limit: null, offset: null });
    expect(out.items.map((h: any) => h.id)).toEqual(["h1", "h2"]);
    expect(out.pagination).toMatchObject({ total: 3, returned: 2, hasNext: true, nextOffset: 2 });
  });

  it("puts everything in one page when no limit was asked for", async () => {
    const out = await getHistories({}, ctxWith(serving(named(["alpha", "beta"]))));
    expect(out.items).toHaveLength(2);
    expect(out.pagination).toMatchObject({ total: 2, returned: 2, hasNext: false, hasPrevious: false });
  });

  it("filters the exact name, and counts what matched", async () => {
    const out = await getHistories({ name: "alpha" }, ctxWith(serving(named(["alpha", "beta", "alpha"]))));
    expect(out.items.map((h: any) => h.id)).toEqual(["h1", "h3"]);
    expect(out.pagination.total).toBe(2);
  });

  it("matches the whole name, not part of it, because bioblend compares for equality", async () => {
    const client = serving(named(["alpha", "beta"]));
    expect((await getHistories({ name: "alp" }, ctxWith(client))).items).toEqual([]);
    expect((await getHistories({ name: "Alpha" }, ctxWith(client))).items).toEqual([]);
  });

  it("is a filter for the histories named \"\", of which there are none", async () => {
    // bioblend filters whenever the parameter is not None, so an empty name matches
    // nothing rather than everything.
    const client = serving(named(["alpha", "beta"]));
    expect((await getHistories({ name: "" }, ctxWith(client))).items).toEqual([]);
    expect((await getHistories({}, ctxWith(client))).items).toHaveLength(2);
  });

  it("ends a filtered walk rather than offering a page that is not there", async () => {
    // The count is a count, so there is nothing to guess about and nothing to carry
    // beside the result: a filtered page reports where it sits in the matches.
    const client = serving(named(["alpha", "beta", "alpha", "alpha"]));
    let offset = 0;
    const seen: string[] = [];
    for (let page = 0; page < 5; page += 1) {
      const out = await getHistories({ name: "alpha", limit: 2, offset }, ctxWith(client));
      seen.push(...out.items.map((h: any) => h.id));
      if (!out.pagination.hasNext) {
        expect(seen).toEqual(["h1", "h3", "h4"]);
        return;
      }
      offset = out.pagination.nextOffset!;
    }
    throw new Error("a filtered walk never reached its last page");
  });

  it("says the same thing about a page whether or not it was projected by its own run", async () => {
    // Nothing lives outside the value, so a result that was copied, serialised or
    // handed on projects to exactly what it projected the first time.
    const client = serving(named(["alpha", "beta", "alpha"]));
    const out = await getHistories({ name: "alpha", limit: 1 }, ctxWith(client));
    const copy = JSON.parse(JSON.stringify(out));
    expect(getHistoriesOp.project!(copy, { name: "alpha", limit: 1 } as never)).toEqual(
      getHistoriesOp.project!(out, { name: "alpha", limit: 1 } as never),
    );
    expect(copy.pagination).toMatchObject({ total: 2, hasNext: true, nextOffset: 1 });
  });

  /**
   * The Python tool validates nothing here, and bioblend drops a falsy limit, so
   * neither a zero limit nor a negative offset is an error on that side. They are
   * not errors on this one either -- a refusal only this surface makes is a call an
   * agent can make against one server and not the other.
   */
  it("takes the windows Python takes, including the silly ones", async () => {
    const client = serving(named(["alpha", "beta", "gamma"]));
    // limit 0: bioblend omits the parameter, so the answer is everything.
    const zero = await getHistories({ limit: 0 }, ctxWith(client));
    expect(zero.items).toHaveLength(3);
    // A negative offset counts from the end, exactly as Python's slice does with the
    // same numbers: [-2:-2+5] over three histories is the last two.
    const negative = await getHistories({ offset: -2, limit: 5 }, ctxWith(client));
    expect(negative.items.map((h: any) => h.id)).toEqual(["h2", "h3"]);
    // And a limit bigger than the account is simply the whole account.
    expect((await getHistories({ limit: 5000 }, ctxWith(client))).items).toHaveLength(3);
  });

  it("handles a user with no histories", async () => {
    const out = await getHistories({}, ctxWith(serving([])));
    expect(out.items).toEqual([]);
    expect(out.pagination).toMatchObject({ total: 0, hasNext: false });
  });

  /**
   * Like the Python tool, this one has no ceiling and no output budget: whatever
   * Galaxy returns for the window asked for is what goes out. Pinned so that
   * nobody adds a bound here without adding it on the other surface too.
   */
  it("returns the page whole, because the Python tool does not budget this one", async () => {
    const record = (n: number) => ({
      id: n.toString(16).padStart(16, "0"),
      name: `ゲノム解析パイプライン ${n}`.repeat(6),
      annotation: "注釈。".repeat(60),
      count: 42,
      update_time: "2026-09-02T08:41:10.000000",
    });
    const rows = Array.from({ length: 300 }, (_, k) => record(k + 1));
    const input = { limit: 300, offset: 0 };
    const out = await getHistories(input, ctxWith(serving(rows)));
    const result = await runWithEnvelope(getHistoriesOp as never, input as never, ctxWith(serving(rows)));
    expect((result.data as { items: unknown[] }).items).toHaveLength(300);
    expect(result.pagination?.trimmedForSize).toBeUndefined();
    expect(mcpPayloadBytes(getHistoriesOp, out, input)).toBeGreaterThan(OUTPUT_BUDGET_BYTES);
  });
});
