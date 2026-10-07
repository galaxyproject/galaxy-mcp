import { describe, it, expect } from "vitest";
import {
  CONTENTS_ORDERS,
  getHistoryContentsOp,
  getHistoryContents,
} from "../../src/operations/get-history-contents";
import { mockClient } from "../util/mock-client";
import { mcpPayloadBytes } from "../util/byte-budget";
import { runWithEnvelope } from "../../src/operations/registry";
import { OUTPUT_BUDGET_BYTES } from "../../src/operations/pagination";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

/**
 * A history contents index entry with every field Galaxy 26.1's summary
 * serializer emits, taken from HDASummary in the published bindings. It is worth
 * being exact about: a measurement taken without `dataset_id`, `genome_build`,
 * `object_store_id`, `quota_source_label` and `type_id` says a default page is
 * comfortable when it is 56,856 bytes on the wire.
 */
const item = (hid: number, state: { deleted?: boolean; visible?: boolean; name?: string } = {}) => ({
  id: hid.toString(16).padStart(16, "b"),
  hid,
  name: state.name ?? `trimmed_reads_${hid}.fastqsanger.gz`,
  history_id: "aaaaaaaaaaaaaaa1",
  history_content_type: "dataset",
  type: "file",
  type_id: `dataset-${hid.toString(16).padStart(16, "b")}`,
  dataset_id: hid.toString(16).padStart(16, "d"),
  extension: "fastqsanger.gz",
  genome_build: "?",
  object_store_id: "files1",
  quota_source_label: null,
  state: "ok",
  deleted: state.deleted ?? false,
  visible: state.visible ?? true,
  purged: false,
  create_time: "2026-09-18T09:12:44.123456",
  update_time: "2026-09-18T09:19:02.654321",
  tags: ["trimmed", "paired"],
  url: `/api/histories/aaaaaaaaaaaaaaa1/contents/${hid.toString(16).padStart(16, "b")}`,
});

/**
 * Galaxy's contents index as the stats media type answers it: filtered by the q/qv
 * pairs, sorted by `order`, windowed by limit and offset, and counted. `asked`
 * collects each request's query and headers.
 */
const serving = (rows: ReturnType<typeof item>[], asked: any[] = []) =>
  mockClient({
    GET: (path, init) => {
      expect(path).toBe("/api/histories/{history_id}/contents");
      const { query, header } = init.params;
      asked.push({ query, header, historyId: init.params.path.history_id });
      const where = Object.fromEntries(query.q.map((field: string, k: number) => [field, query.qv[k]]));
      let matching = rows.filter(
        (row) => !(where.deleted === "False" && row.deleted) && !(where.visible === "True" && !row.visible),
      );
      const [field, direction] = query.order.split("-") as [keyof typeof matching[0], string];
      matching = [...matching].sort((a, b) => (a[field]! < b[field]! ? -1 : a[field]! > b[field]! ? 1 : 0));
      if (direction === "dsc") matching.reverse();
      return {
        data: {
          contents: matching.slice(query.offset, query.offset + query.limit),
          stats: { total_matches: matching.length },
        },
        response: { status: 200 },
      };
    },
  });

/** `total` items, every fourth one deleted and every fourth hidden, offset by one. */
const history = (total: number) =>
  Array.from({ length: total }, (_, k) => item(k + 1, { deleted: k % 4 === 1, visible: k % 4 !== 2 }));

/** How many of `history(total)` survive the default filters. */
const live = (total: number) =>
  Array.from({ length: total }, (_, k) => k).filter((k) => k % 4 !== 1 && k % 4 !== 2).length;

describe("get_history_contents", () => {
  it("asks Galaxy for one filtered, sorted window with its count", async () => {
    const asked: any[] = [];
    await getHistoryContents({ historyId: "h1", limit: 7, offset: 14 }, ctxWith(serving(history(40), asked)));
    expect(asked).toEqual([
      {
        historyId: "h1",
        query: {
          v: "dev",
          limit: 7,
          offset: 14,
          order: "hid-asc",
          q: ["deleted", "visible"],
          qv: ["False", "True"],
        },
        header: { accept: "application/vnd.galaxy.history.contents.stats+json" },
      },
    ]);
  });

  it("drops a filter the caller widened", async () => {
    const asked: any[] = [];
    await getHistoryContents({ historyId: "h1", deleted: true, visible: false }, ctxWith(serving([], asked)));
    expect(asked[0].query).toMatchObject({ q: [], qv: [] });
  });

  it("reports the total Galaxy counted, not the page it sent", async () => {
    const out = await getHistoryContents({ historyId: "h1", limit: 10 }, ctxWith(serving(history(500))));
    expect(out.items).toHaveLength(10);
    expect(out.pagination).toMatchObject({ total: live(500), hasNext: true, nextOffset: 10 });
  });

  it("passes each documented order to Galaxy as named", async () => {
    for (const order of CONTENTS_ORDERS) {
      const asked: any[] = [];
      await getHistoryContents({ historyId: "h1", order }, ctxWith(serving([], asked)));
      expect(asked[0].query.order).toBe(order);
    }
  });

  it("refuses any other order before it asks Galaxy anything", async () => {
    const asked: any[] = [];
    await expect(
      getHistoryContents({ historyId: "h1", order: "hid" as never }, ctxWith(serving([], asked))),
    ).rejects.toThrow(
      "order must be one of hid-asc, hid-dsc, create_time-asc, create_time-dsc, update_time-asc, update_time-dsc, name-asc, name-dsc, extension-asc, extension-dsc, size-asc, size-dsc (got 'hid')",
    );
    expect(asked).toEqual([]);
    expect(getHistoryContentsOp.input.order.safeParse("nonsense-asc").success).toBe(false);
  });

  it("says what each item is when Galaxy did not", async () => {
    const untyped = ({ history_content_type: _, ...row }: ReturnType<typeof item>) => row;
    const rows = [untyped(item(1)), { ...untyped(item(2)), collection_type: "list" }];
    const out = await getHistoryContents({ historyId: "h1" }, ctxWith(serving(rows as never)));
    expect(out.items.map((i: any) => i.history_content_type)).toEqual(["dataset", "dataset_collection"]);
  });

  it("leaves dataset_id out of the rows a caller reads, and keeps id", async () => {
    const result = await runWithEnvelope(
      getHistoryContentsOp as never,
      { historyId: "h1" } as never,
      ctxWith(serving([item(1)])),
    );
    const [row] = (result.data as { contents: Record<string, unknown>[] }).contents;
    expect(row).not.toHaveProperty("dataset_id");
    expect(row.id).toBe(item(1).id);
  });

  describe("a window that is not a window", () => {
    it("refuses a limit below one, in the Python server's words", async () => {
      const client = serving(history(8));
      await expect(getHistoryContents({ historyId: "h1", limit: 0 }, ctxWith(client))).rejects.toThrow(
        "limit must be at least 1 (got 0)",
      );
      await expect(getHistoryContents({ historyId: "h1", limit: -5 }, ctxWith(client))).rejects.toThrow(
        "limit must be at least 1 (got -5)",
      );
    });

    it("refuses a negative offset, in the Python server's words", async () => {
      const client = serving(history(8));
      await expect(
        getHistoryContents({ historyId: "h1", limit: 10, offset: -1 }, ctxWith(client)),
      ).rejects.toThrow("offset must be 0 or greater (got -1)");
    });

    // The third refusal in the rule, and the one the release notes left out: a window
    // has to be a whole number of rows. On the wire nothing changed -- both servers
    // declare these parameters as integers and refuse a fraction at the schema, the
    // other one with pydantic's "Input should be a valid integer, got a number with a
    // fractional part" -- so this is the library path, where it used to return a page.
    // The two sentences are written out rather than imported, so a change to either
    // has to be made here as well as in the notes that quote them.
    it("refuses a limit that is not a whole number of rows", async () => {
      const client = serving(history(8));
      await expect(getHistoryContents({ historyId: "h1", limit: 1.5 }, ctxWith(client))).rejects.toThrow(
        "limit must be a whole number (got 1.5)",
      );
    });

    it("refuses a fractional offset with the sentence it refuses a negative one with", async () => {
      const client = serving(history(8));
      await expect(
        getHistoryContents({ historyId: "h1", limit: 10, offset: 0.5 }, ctxWith(client)),
      ).rejects.toThrow("offset must be 0 or greater (got 0.5)");
    });

    it("refuses before it asks Galaxy anything", async () => {
      let asked = 0;
      const counting = mockClient({
        GET: () => {
          asked += 1;
          return { data: { contents: [], stats: { total_matches: 0 } }, response: { status: 200 } };
        },
      });
      await expect(getHistoryContents({ historyId: "h1", limit: 0 }, ctxWith(counting))).rejects.toThrow();
      expect(asked).toBe(0);
    });

    it("still takes a limit no ceiling would allow, because neither server caps this one", async () => {
      const out = await getHistoryContents({ historyId: "h1", limit: 5000 }, ctxWith(serving(history(8))));
      expect(out.items).toHaveLength(live(8));
    });

    it("takes the smallest window there is", async () => {
      const out = await getHistoryContents({ historyId: "h1", limit: 1, offset: 0 }, ctxWith(serving(history(8))));
      expect(out.items).toHaveLength(1);
      expect(out.pagination).toMatchObject({ limit: 1, offset: 0, hasNext: true, nextOffset: 1 });
    });
  });

  it("handles an empty history", async () => {
    const out = await getHistoryContents({ historyId: "h1" }, ctxWith(serving([])));
    expect(out.items).toEqual([]);
    expect(out.pagination).toMatchObject({ total: 0, returned: 0, hasNext: false });
  });

  it("reports a short final page as the last one", async () => {
    const total = live(500);
    const out = await getHistoryContents(
      { historyId: "h1", limit: 10, offset: total - 3 },
      ctxWith(serving(history(500))),
    );
    expect(out.items).toHaveLength(3);
    expect(out.pagination).toMatchObject({ hasNext: false, hasPrevious: true });
  });

  /**
   * A default page of 100 full records is 56,856 bytes on the wire, past what a client
   * passes through. Both surfaces cut it to the shared output budget, and the next page
   * starts at the first item this one did not return.
   */
  it("cuts a page over the output budget and continues from the first item it left out", async () => {
    const input = { historyId: "h1", limit: 100, offset: 0 };
    const out = await getHistoryContents(input, ctxWith(serving(history(500))));
    expect(mcpPayloadBytes(getHistoryContentsOp, out, input)).toBeGreaterThan(OUTPUT_BUDGET_BYTES);
    const result = await runWithEnvelope(getHistoryContentsOp as never, input as never, ctxWith(serving(history(500))));
    const contents = (result.data as { contents: { hid: number }[] }).contents;
    const returned = contents.length;
    expect(returned).toBeGreaterThan(0);
    expect(returned).toBeLessThan(100);
    expect(new TextEncoder().encode(JSON.stringify(result)).length).toBeLessThanOrEqual(OUTPUT_BUDGET_BYTES);
    expect(result.pagination).toMatchObject({ total_items: live(500), next_offset: returned, has_next: true });
    expect(result.pagination?.helper_text).toContain("cut short to fit the output budget");
  });
});
