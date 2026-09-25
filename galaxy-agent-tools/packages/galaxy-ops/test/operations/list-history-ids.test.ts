import { describe, it, expect } from "vitest";
import { listHistoryIdsOp, listHistoryIds } from "../../src/operations/list-history-ids";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

/** A Galaxy history index entry is an encoded id and a user-typed name. */
const histories = (n: number) =>
  Array.from({ length: n }, (_, k) => ({
    id: (k + 1).toString(16).padStart(16, "a"),
    name: `Imported: RNA-seq analysis of Arabidopsis root tissue ${k + 1} (copy)`,
  }));

const serving = (n: number) => mockClient({ GET: () => ({ data: histories(n), response: { status: 200 } }) });

describe("list_history_ids", () => {
  it("returns id/name pairs", async () => {
    const out = await listHistoryIds({}, ctxWith(serving(1)));
    expect(out.items).toEqual([{ id: histories(1)[0].id, name: histories(1)[0].name }]);
    expect(listHistoryIdsOp.name).toBe("list_history_ids");
  });

  it("returns the default page of 100 and points at the next one", async () => {
    const out = await listHistoryIds({}, ctxWith(serving(250)));
    expect(out.items).toHaveLength(100);
    expect(out.pagination).toMatchObject({ total: 250, returned: 100, limit: 100, hasNext: true, nextOffset: 100 });
  });

  it("honours an explicit page", async () => {
    const out = await listHistoryIds({ limit: 10, offset: 240 }, ctxWith(serving(250)));
    expect(out.items).toHaveLength(10);
    expect(out.items[0].id).toBe(histories(250)[240].id);
    expect(out.pagination).toMatchObject({ hasNext: false, hasPrevious: true });
  });

  it("rejects a limit over the ceiling the Python tool sets, instead of clamping it", async () => {
    await expect(listHistoryIds({ limit: 501 }, ctxWith(serving(10)))).rejects.toThrow(/at most 500/);
  });

  it("handles a user with no histories", async () => {
    const out = await listHistoryIds({}, ctxWith(serving(0)));
    expect(out.items).toEqual([]);
    expect(out.pagination).toMatchObject({ total: 0, hasNext: false });
  });

});
