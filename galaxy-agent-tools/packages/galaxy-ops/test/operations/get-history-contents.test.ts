import { describe, it, expect } from "vitest";
import { getHistoryContentsOp, getHistoryContents } from "../../src/operations/get-history-contents";
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
 * The contents index as Galaxy answers it with no query parameters, which is what
 * `show_history(contents=True)` asks for on the other surface: everything, in hid
 * order, deleted and hidden items included.
 */
const serving = (total: number, seen?: (query: any) => void) =>
  mockClient({
    GET: (path, init) => {
      expect(path).toBe("/api/histories/{history_id}/contents");
      seen?.(init.params.query);
      return {
        data: Array.from({ length: total }, (_, k) =>
          item(k + 1, { deleted: k % 4 === 1, visible: k % 4 !== 2 }),
        ),
        response: { status: 200 },
      };
    },
  });

/** How many of `serving(total)` survive the default filters. */
const live = (total: number) =>
  Array.from({ length: total }, (_, k) => k).filter((k) => k % 4 !== 1 && k % 4 !== 2).length;

describe("get_history_contents", () => {
  it("lists a history's contents by id", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(init.params.path.history_id).toBe("h1");
        return { data: [item(1)], response: { status: 200 } };
      },
    });
    const out = await getHistoryContents({ historyId: "h1" }, ctxWith(client));
    expect(out.items).toHaveLength(1);
  });

  it("asks Galaxy for the whole index and windows it here, the way the Python tool does", async () => {
    let query: any;
    const out = await getHistoryContents({ historyId: "h1" }, ctxWith(serving(500, (q) => (query = q))));
    // No filters, no order, no window on the wire: Galaxy is asked for everything.
    expect(query).toBeUndefined();
    expect(out.items).toHaveLength(100);
    expect(out.pagination.total).toBe(live(500));
  });

  it("excludes deleted and hidden items by default, and includes them when asked", async () => {
    const byDefault = await getHistoryContents({ historyId: "h1" }, ctxWith(serving(20)));
    expect(byDefault.items.every((entry: any) => !entry.deleted && entry.visible)).toBe(true);
    expect(byDefault.pagination.total).toBe(live(20));

    const everything = await getHistoryContents(
      { historyId: "h1", deleted: true, visible: false },
      ctxWith(serving(20)),
    );
    expect(everything.pagination.total).toBe(20);
  });

  it("reports a real total, because it did the counting", async () => {
    const out = await getHistoryContents({ historyId: "h1", limit: 5, offset: 5 }, ctxWith(serving(100)));
    expect(out.pagination).toMatchObject({
      total: live(100),
      returned: 5,
      offset: 5,
      hasNext: true,
      hasPrevious: true,
      nextOffset: 10,
      previousOffset: 0,
    });
  });

  /**
   * The ordering rules are the Python tool's, which are not Galaxy's: it tests the
   * prefix of `order` and reverses only on `-dsc`, so "hid" is ascending where
   * Galaxy's index would answer descending, and an order it does not recognise is
   * hid ascending rather than a 400.
   */
  it("sorts by the prefix of order and reverses only on -dsc", async () => {
    const hids = async (order?: string) =>
      (await getHistoryContents({ historyId: "h1", order, limit: 4 }, ctxWith(serving(40)))).items.map(
        (entry: any) => entry.hid,
      );
    const ascending = await hids("hid-asc");
    expect(await hids("hid")).toEqual(ascending);
    expect(await hids("nonsense-asc")).toEqual(ascending);
    expect(await hids()).toEqual(ascending);
    // -dsc is the only suffix that reverses, and it reverses the whole set before
    // the window, so the first page is the highest hids rather than the lowest.
    const descending = await hids("hid-dsc");
    expect(descending[0]).toBeGreaterThan(descending[3]!);
    expect(descending[0]).toBeGreaterThan(ascending[3]!);
  });

  it("sorts by name and by time when asked", async () => {
    const named = mockClient({
      GET: () => ({
        data: [item(1, { name: "zeta" }), item(2, { name: "alpha" }), item(3, { name: "mu" })],
        response: { status: 200 },
      }),
    });
    const byName = await getHistoryContents({ historyId: "h1", order: "name-asc" }, ctxWith(named));
    expect(byName.items.map((entry: any) => entry.name)).toEqual(["alpha", "mu", "zeta"]);
    const byNameDown = await getHistoryContents({ historyId: "h1", order: "name-dsc" }, ctxWith(named));
    expect(byNameDown.items.map((entry: any) => entry.name)).toEqual(["zeta", "mu", "alpha"]);
  });

  it("says what each item is when Galaxy did not", async () => {
    const client = mockClient({
      GET: () => ({
        data: [
          { hid: 1, name: "a dataset" },
          { hid: 2, name: "a collection", collection_type: "list" },
          { hid: 3, name: "explicit", history_content_type: "dataset_collection" },
        ],
        response: { status: 200 },
      }),
    });
    const out = await getHistoryContents({ historyId: "h1" }, ctxWith(client));
    expect(out.items.map((entry: any) => entry.history_content_type)).toEqual([
      "dataset",
      "dataset_collection",
      "dataset_collection",
    ]);
  });

  it("honours an explicit page", async () => {
    const out = await getHistoryContents({ historyId: "h1", limit: 5, offset: 20 }, ctxWith(serving(500)));
    expect(out.items).toHaveLength(5);
    expect(out.pagination).toMatchObject({ limit: 5, offset: 20 });
  });

  /**
   * The Python tool validates nothing here either: limit 0 slices to nothing and a
   * negative limit or offset is whatever Python's slice does with it. Refusing
   * these would be a refusal only this surface makes.
   */
  it("takes the windows Python takes, including the silly ones", async () => {
    const client = serving(8);
    expect((await getHistoryContents({ historyId: "h1", limit: 0 }, ctxWith(client))).items).toEqual([]);
    expect((await getHistoryContents({ historyId: "h1", limit: -5 }, ctxWith(client))).items).toEqual([]);
    const fromTheEnd = await getHistoryContents({ historyId: "h1", offset: -1 }, ctxWith(client));
    expect(fromTheEnd.items).toHaveLength(1);
    expect((await getHistoryContents({ historyId: "h1", limit: 5000 }, ctxWith(client))).items).toHaveLength(
      live(8),
    );
  });

  it("handles an empty history", async () => {
    const out = await getHistoryContents({ historyId: "h1" }, ctxWith(serving(0)));
    expect(out.items).toEqual([]);
    expect(out.pagination).toMatchObject({ total: 0, returned: 0, hasNext: false });
  });

  it("reports a short final page as the last one", async () => {
    const total = live(500);
    const out = await getHistoryContents({ historyId: "h1", limit: 10, offset: total - 3 }, ctxWith(serving(500)));
    expect(out.items).toHaveLength(3);
    expect(out.pagination).toMatchObject({ hasNext: false, hasPrevious: true });
  });

  /**
   * Neither surface bounds this one. The Python tool has no MAX_PAGE_SIZE entry
   * and does not go through _budgeted_page, so a default page of 100 fat records
   * is whatever size it is -- 56,856 bytes for the records above, past what a
   * client will pass through. Pinned rather than fixed: closing it means changing
   * the Python tool too, and the two have to keep answering the same way.
   */
  it("returns the page whole, because the Python tool does not budget this one either", async () => {
    const input = { historyId: "h1", limit: 100, offset: 0 };
    const out = await getHistoryContents(input, ctxWith(serving(500)));
    const result = await runWithEnvelope(getHistoryContentsOp as never, input as never, ctxWith(serving(500)));
    expect(out.items).toHaveLength(100);
    expect((result.data as { items: unknown[] }).items).toHaveLength(100);
    expect(result.pagination?.trimmedForSize).toBeUndefined();
    expect(mcpPayloadBytes(getHistoryContentsOp, out, input)).toBeGreaterThan(OUTPUT_BUDGET_BYTES);
  });
});
