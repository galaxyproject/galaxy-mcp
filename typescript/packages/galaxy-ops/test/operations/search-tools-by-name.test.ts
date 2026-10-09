import { describe, it, expect } from "vitest";
import { searchToolsByNameOp, searchToolsByName } from "../../src/operations/search-tools-by-name";
import { mockClient } from "../util/mock-client";
import { paginate } from "../../src/operations/pagination";
import { toolIndex } from "../util/tool-fixture";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

const TOOLS = [
  { id: "fastqc_tool", name: "FastQC", description: "Quality control" },
  { id: "trimmomatic", name: "Trimmomatic", description: "Trimming reads" },
  // id-only match fixture: name/description intentionally contain no part of the id
  { id: "toolxyz_internal_id", name: "Read Processor", description: "Processes sequencing reads" },
  { id: "cat1", name: "Concatenate", description: "Join files together" },
];

describe("search_tools_by_name", () => {
  it("sends in_panel=false and filters by name substring (case-insensitive)", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/tools");
        expect(init.params.query.in_panel).toBe(false);
        return { data: TOOLS, response: { status: 200 } };
      },
    });
    const { items: out } = await searchToolsByName({ query: "fastqc" }, ctxWith(client));
    expect(out).toHaveLength(1);
    expect(out[0].id).toBe("fastqc_tool");
  });

  it("matches against id when name/description do not match", async () => {
    const client = mockClient({
      GET: () => ({ data: TOOLS, response: { status: 200 } }),
    });
    // "toolxyz_internal_id" contains "toolxyz" -- the name ("Read Processor") and
    // description ("Processes sequencing reads") have no substring match, so this
    // exercises the id-only branch exclusively
    const { items: out } = await searchToolsByName({ query: "toolxyz" }, ctxWith(client));
    expect(out).toHaveLength(1);
    expect(out[0].id).toBe("toolxyz_internal_id");
  });

  it("matches against description (case-insensitive)", async () => {
    const client = mockClient({
      GET: () => ({ data: TOOLS, response: { status: 200 } }),
    });
    const { items: out } = await searchToolsByName({ query: "join files" }, ctxWith(client));
    expect(out).toHaveLength(1);
    expect(out[0].id).toBe("cat1");
  });

  /**
   * Case folding is Unicode data too, and the two runtimes read different editions of it.
   * The other server was asked directly:
   *
   *   >>> "\u0264".lower() in "RAMS HORN \ua7cb".lower()   -> False
   *
   * because it has no case mapping for the capital rams horn at all, while `toLowerCase`
   * here folds it onto the small one and reports a match nobody else makes.
   */
  it("case-folds a needle the way the other server does, not the way this runtime does", async () => {
    const client = mockClient({
      GET: () => ({
        data: [{ id: "rams_horn", name: "Rams Horn \ua7cb", description: "" }],
        response: { status: 200 },
      }),
    });
    const { items: out } = await searchToolsByName({ query: "\u0264" }, ctxWith(client));
    expect(out).toHaveLength(0);
  });

  /**
   * And the other half of case folding, which a table of mappings does not cover: whether a
   * sigma is at the end of a word. The other server was asked directly:
   *
   *   >>> "\u1c8a\u03a3".lower()          -> '\u1c8a\u03c3'
   *   >>> "\u03c3" in "\u1c8a\u03a3".lower() -> True
   *
   * U+1C8A lowercases to itself on both runtimes, so it is not a mapping difference at all --
   * it is a cased letter here and unassigned there, and this runtime therefore reads the sigma
   * after it as final. A needle of a plain sigma then finds the tool on that server and not on
   * this one. Also pinned end to end as the golden envelope
   * `search_tools_by_name/sigma_after_a_letter_assigned_after_unicode_15`.
   */
  it("finds a sigma the other server does not make final", async () => {
    const client = mockClient({
      GET: () => ({
        data: [{ id: "t", name: "\u1c8a\u03a3", description: "" }],
        response: { status: 200 },
      }),
    });
    const { items: out } = await searchToolsByName({ query: "\u03c3" }, ctxWith(client));
    expect(out).toHaveLength(1);
    expect(out[0]!.id).toBe("t");
  });

  it("returns empty array when no match", async () => {
    const client = mockClient({
      GET: () => ({ data: TOOLS, response: { status: 200 } }),
    });
    const { items: out } = await searchToolsByName({ query: "zzznomatch" }, ctxWith(client));
    expect(out).toHaveLength(0);
  });

  it("throws on 500 error", async () => {
    const client = mockClient({
      GET: () => ({ error: { err_msg: "server error" }, response: { status: 500 } }),
    });
    await expect(searchToolsByName({ query: "anything" }, ctxWith(client))).rejects.toThrow();
  });

  it("project returns message with count and query", () => {
    const paged = paginate([{ id: "t1", name: "Tool" }], { limit: 25, offset: 0, noun: "tools" });
    const msg = searchToolsByNameOp.project!(paged, { query: "tool" } as never);
    expect(msg.message).toBe("Found 1 tools matching 'tool', returning 1");
  });
});

describe("search_tools_by_name paging", () => {
  const serving = (n: number) => mockClient({ GET: () => ({ data: toolIndex(n), response: { status: 200 } }) });

  it("returns the default page of 25 and points at the next", async () => {
    const out = await searchToolsByName({ query: "bwa" }, ctxWith(serving(200)));
    expect(out.items).toHaveLength(25);
    expect(out.pagination).toMatchObject({ total: 200, returned: 25, hasNext: true, nextOffset: 25 });
  });

  it("honours an explicit page", async () => {
    const out = await searchToolsByName({ query: "bwa", limit: 10, offset: 195 }, ctxWith(serving(200)));
    expect(out.items).toHaveLength(5);
    expect(out.pagination).toMatchObject({ hasNext: false, hasPrevious: true });
  });

  it("rejects a limit over the ceiling the Python tool sets, without calling Galaxy", async () => {
    const client = mockClient({ GET: () => { throw new Error("should not reach Galaxy"); } });
    await expect(searchToolsByName({ query: "bwa", limit: 101 }, ctxWith(client))).rejects.toThrow(/at most 100/);
  });

  it("reports a query that matches nothing as an empty last page", async () => {
    const out = await searchToolsByName({ query: "zzz-no-match-zzz" }, ctxWith(serving(200)));
    expect(out.items).toEqual([]);
    expect(out.pagination).toMatchObject({ total: 0, hasNext: false });
  });

});
