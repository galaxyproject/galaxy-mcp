import { describe, it, expect } from "vitest";
import { searchToolsByKeywordsOp, searchToolsByKeywords } from "../../src/operations/search-tools-by-keywords";
import { mockClient } from "../util/mock-client";
import { paginate } from "../../src/operations/pagination";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

// A minimal panel with nested sections and tools
const PANEL = [
  {
    id: "sec_genomics",
    name: "Genomics",
    model_class: "ToolSection",
    elems: [
      // immediate match: name contains "fastqc"
      { id: "fastqc_tool", name: "FastQC", description: "Quality control for FASTQ" },
      // _label id: must be skipped
      { id: "genomics_label", name: "Genomics Label", description: "" },
    ],
  },
  // a nested section inside another section
  {
    id: "sec_assembly",
    name: "Assembly",
    model_class: "ToolSection",
    elems: [
      {
        id: "sec_inner",
        name: "Inner",
        model_class: "ToolSection",
        elems: [
          // extensions-only match: neither name nor description contains "bam"
          { id: "samtools_view", name: "SAMtools View", description: "Convert alignment files" },
        ],
      },
    ],
  },
];

describe("search_tools_by_keywords", () => {
  it("keeps a real tool whose id ends in _label, the way the panel op does", async () => {
    // Galaxy says what a node is with model_class; get_tool_panel already trusts it,
    // and a tool this one dropped would be visible in the panel and unfindable here.
    const client = mockClient({
      GET: (path: string) => {
        expect(path).toBe("/api/tools");
        return {
          data: [
            {
              id: "sec_x",
              name: "X",
              model_class: "ToolSection",
              elems: [
                { id: "add_column_label", name: "Add column label", description: "Label a column", model_class: "Tool" },
                { id: "x_label", name: "Divider", model_class: "ToolSectionLabel" },
              ],
            },
          ],
          response: { status: 200 },
        };
      },
    });
    const out = await searchToolsByKeywords({ keywords: ["column"] }, ctxWith(client));
    expect(out.items.map((m) => m.id)).toEqual(["add_column_label"]);
  });

  it("returns immediate name match without fetching tool details", async () => {
    let detailFetchCount = 0;
    const client = mockClient({
      GET: (path, init) => {
        if (path === "/api/tools" && init?.params?.query?.in_panel === true) {
          return { data: PANEL, response: { status: 200 } };
        }
        // tool detail fetch for extensions matching
        detailFetchCount++;
        return { data: { id: init.params.path.tool_id, inputs: [] }, response: { status: 200 } };
      },
    });
    const { items: out } = await searchToolsByKeywords({ keywords: ["fastqc"] }, ctxWith(client));
    const ids = out.map((t) => t.id);
    expect(ids).toContain("fastqc_tool");
  });

  it("never fetches details for a _label id, but keeps one that matched by name", async () => {
    // Where the Python tool checks the suffix, and therefore where this one does:
    // a divider that matched on name is a result on both surfaces, and one that
    // did not is not worth a request. get_tool_panel answers differently about the
    // same node, because it copies a different helper on the other side.
    const fetched: string[] = [];
    const client = mockClient({
      GET: (path, init) => {
        if (path === "/api/tools" && init?.params?.query?.in_panel === true) {
          return { data: PANEL, response: { status: 200 } };
        }
        fetched.push(init.params.path.tool_id);
        return { data: { id: init.params.path.tool_id, inputs: [] }, response: { status: 200 } };
      },
    });
    // The panel gets a divider that does NOT match the keyword, which is the only
    // way through to the suffix guard: genomics_label matches "label" by name and is
    // answered from the panel alone, so on its own it proves nothing about fetching.
    const panel = [
      ...PANEL,
      { id: "quiet_label", name: "Divider", description: "", model_class: "ToolSectionLabel" },
    ];
    const withDivider = mockClient({
      GET: (path: string, init: any) => {
        if (path === "/api/tools" && init?.params?.query?.in_panel === true) {
          return { data: panel, response: { status: 200 } };
        }
        fetched.push(init.params.path.tool_id);
        return { data: { id: init.params.path.tool_id, inputs: [] }, response: { status: 200 } };
      },
    });
    const { items: out } = await searchToolsByKeywords({ keywords: ["label"] }, ctxWith(withDivider));
    expect(out.map((t) => t.id)).toContain("genomics_label");
    expect(out.map((t) => t.id)).not.toContain("quiet_label");
    expect(fetched).not.toContain("quiet_label");
    // And the guard is the reason, not luck: everything else that missed the
    // keyword did get a request.
    expect(fetched).toContain("samtools_view");
  });

  it("matches extensions-only tools via detail fetch", async () => {
    let detailFetchCount = 0;
    const client = mockClient({
      GET: (path, init) => {
        if (path === "/api/tools" && init?.params?.query?.in_panel === true) {
          return { data: PANEL, response: { status: 200 } };
        }
        // tool detail fetch -- samtools_view has bam extension input, nothing else does
        detailFetchCount++;
        const toolId = init?.params?.path?.tool_id;
        if (toolId === "samtools_view") {
          return {
            data: {
              id: "samtools_view",
              inputs: [{ extensions: ["bam", "sam"] }],
            },
            response: { status: 200 },
          };
        }
        return { data: { id: toolId, inputs: [] }, response: { status: 200 } };
      },
    });
    const { items: out } = await searchToolsByKeywords({ keywords: ["bam"] }, ctxWith(client));
    const ids = out.map((t) => t.id);
    // samtools_view has no "bam" in name or description -- it must match via extension detail fetch
    expect(ids).toContain("samtools_view");
    // assert the detail endpoint was actually called (extensions branch is exercised)
    expect(detailFetchCount).toBeGreaterThan(0);
  });

  it("returns slim objects with id/name/description/versions", async () => {
    const client = mockClient({
      GET: (path, init) => {
        if (path === "/api/tools" && init?.params?.query?.in_panel === true) {
          return { data: PANEL, response: { status: 200 } };
        }
        return { data: { id: init?.params?.path?.tool_id, inputs: [] }, response: { status: 200 } };
      },
    });
    const { items: out } = await searchToolsByKeywords({ keywords: ["fastqc"] }, ctxWith(client));
    expect(out[0]).toHaveProperty("id");
    // name/description/versions are allowed to be present (optional)
    expect(Object.keys(out[0])).not.toContain("elems");
  });

  it("project returns message with match count", () => {
    const paged = paginate([{ id: "t1" }, { id: "t2" }], { limit: 50, offset: 0, noun: "tools" });
    const msg = searchToolsByKeywordsOp.project!(paged, { keywords: ["fastqc"] } as never);
    expect(msg.message).toBe("2 of 2 tool(s) matching keywords");
  });
});

describe("search_tools_by_keywords paging", () => {
  // Every tool matches on name, so nothing takes the detail-fetch path. Used
  // only where the test is about the window arithmetic itself.
  const byName = (n: number) =>
    mockClient({
      GET: () => ({
        data: [
          {
            id: "matching_section",
            name: "Matching",
            elems: Array.from({ length: n }, (_, k) => ({
              id: `toolshed.g2.bx.psu.edu/repos/iuc/bwa_mem_${k}/bwa_mem_${k}/0.7.17+galaxy1`,
              name: `Map with BWA-MEM variant ${k}`,
              description: "map medium and long reads (> 100 bp) against a reference genome",
              versions: ["0.7.17+galaxy1", "0.7.17+galaxy0"],
            })),
          },
        ],
        response: { status: 200 },
      }),
    });

  /**
   * A panel where NO tool matches on name or description, so every one of them
   * goes through the detail fetch and the match is decided by input extensions.
   * This is the path the ordering claim is actually about.
   */
  const byExtension = (n: number, matching: (k: number) => boolean) =>
    mockClient({
      GET: (path, init) => {
        if (path === "/api/tools") {
          return {
            data: [
              {
                id: "s",
                name: "Section",
                elems: Array.from({ length: n }, (_, k) => ({
                  id: `tool_${String(k).padStart(3, "0")}`,
                  name: `Opaque tool ${k}`,
                  description: "does something unrelated to the query",
                  versions: ["1.0"],
                })),
              },
            ],
            response: { status: 200 },
          };
        }
        const id: string = init.params.path.tool_id;
        const k = Number(id.slice("tool_".length));
        const answer = {
          data: { id, inputs: matching(k) ? [{ extensions: ["bam"] }] : [{ extensions: ["txt"] }] },
          response: { status: 200 },
        };
        // Deliberately out of order: within each group of ten in flight, the last
        // request started is the first to answer. Details that all resolve at once
        // settle in the order they were made, which is the one order that cannot
        // tell an indexed write from a push.
        return new Promise((resolve) => setTimeout(() => resolve(answer), (9 - (k % 10)) * 2));
      },
    });

  it("returns the default page of 50 and points at the next", async () => {
    const out = await searchToolsByKeywords({ keywords: ["bwa"] }, ctxWith(byName(400)));
    expect(out.items).toHaveLength(50);
    expect(out.pagination).toMatchObject({ total: 400, returned: 50, hasNext: true, nextOffset: 50 });
  });

  it("honours an explicit page", async () => {
    const out = await searchToolsByKeywords({ keywords: ["bwa"], limit: 10, offset: 395 }, ctxWith(byName(400)));
    expect(out.items).toHaveLength(5);
    expect(out.pagination).toMatchObject({ hasNext: false, hasPrevious: true });
  });

  it("rejects a limit over the ceiling the Python tool sets, without calling Galaxy", async () => {
    const client = mockClient({ GET: () => { throw new Error("should not reach Galaxy"); } });
    await expect(searchToolsByKeywords({ keywords: ["bwa"], limit: 201 }, ctxWith(client))).rejects.toThrow(
      /at most 200/,
    );
  });

  it("reports a keyword that matches none of a full panel as an empty last page", async () => {
    // 400 real tools, none of which match on name, description or extension.
    const out = await searchToolsByKeywords({ keywords: ["zzz-no-match-zzz"] }, ctxWith(byExtension(400, () => false)));
    expect(out.items).toEqual([]);
    expect(out.pagination).toMatchObject({ total: 0, hasNext: false });
  });

  it("keeps the extension-matched set in panel order across calls and pages", async () => {
    // Every tool takes the detail-fetch path, which is where concurrency could
    // reorder the results; a page walk over a reordered set skips and duplicates.
    // The fixture answers out of order on purpose, so this fails if the results are
    // collected as they complete rather than written by index.
    const matching = (k: number) => k % 3 === 0;
    const walk = async () => {
      const seen: string[] = [];
      let offset = 0;
      for (;;) {
        const out = await searchToolsByKeywords(
          { keywords: ["bam"], limit: 40, offset },
          ctxWith(byExtension(300, matching)),
        );
        seen.push(...out.items.map((t) => t.id));
        if (out.pagination.nextOffset === undefined) break;
        offset = out.pagination.nextOffset;
      }
      return seen;
    };
    const expected = Array.from({ length: 300 }, (_, k) => k)
      .filter(matching)
      .map((k) => `tool_${String(k).padStart(3, "0")}`);
    expect(await walk()).toEqual(expected);
    expect(await walk()).toEqual(expected);
  });

});
