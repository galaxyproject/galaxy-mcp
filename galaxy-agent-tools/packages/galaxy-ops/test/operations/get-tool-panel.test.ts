import { describe, it, expect } from "vitest";
import {
  getToolPanelOp,
  getToolPanel,
  type ToolPanelOverview,
  type ToolPanelSection,
} from "../../src/operations/get-tool-panel";
import { mockClient } from "../util/mock-client";
import { toolPanel } from "../util/tool-fixture";
import { DEFAULT_POLL } from "../../src/context";
import { GalaxyNotFoundError } from "../../src/errors";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

const PANEL = [
  { id: "section1", name: "Genomics", model_class: "ToolSection", elems: [{ id: "fastqc" }] },
  { id: "section2", name: "Assembly", model_class: "ToolSection", elems: [] },
];

describe("get_tool_panel", () => {
  it("fetches the panel via in_panel=true and summarises it into sections", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/tools");
        expect(init.params.query.in_panel).toBe(true);
        return { data: PANEL, response: { status: 200 } };
      },
    });
    const out = (await getToolPanel({}, ctxWith(client))) as ToolPanelOverview;
    expect(out.entries).toEqual([
      { id: "section1", name: "Genomics", type: "section", tool_count: 1 },
      { id: "section2", name: "Assembly", type: "section", tool_count: 0 },
    ]);
  });

  it("throws GalaxyNotFoundError on 404", async () => {
    const client = mockClient({
      GET: () => ({ error: { err_msg: "not found" }, response: { status: 404 } }),
    });
    await expect(getToolPanel({}, ctxWith(client))).rejects.toBeInstanceOf(GalaxyNotFoundError);
  });

  it("project counts the page against the total", async () => {
    const client = mockClient({ GET: () => ({ data: PANEL, response: { status: 200 } }) });
    const out = await getToolPanel({}, ctxWith(client));
    expect(getToolPanelOp.project!(out, {} as never).message).toBe("2 of 2 tool panel entries");
  });
});

describe("get_tool_panel paging and drill-in", () => {
  const serving = (sections: number, tools: number) =>
    mockClient({ GET: () => ({ data: toolPanel(sections, tools), response: { status: 200 } }) });

  it("returns the default page of 100 entries and points at the next", async () => {
    const out = (await getToolPanel({}, ctxWith(serving(250, 5)))) as ToolPanelOverview;
    expect(out.entries).toHaveLength(100);
    expect(out.pagination).toMatchObject({ total: 250, returned: 100, hasNext: true, nextOffset: 100 });
  });

  it("counts only runnable tools in a section, not its dividers", async () => {
    const out = (await getToolPanel({}, ctxWith(serving(1, 7)))) as ToolPanelOverview;
    expect(out.entries[0]).toMatchObject({ type: "section", tool_count: 7 });
  });

  it("lists a tool sitting outside any section as a tool entry", async () => {
    const client = mockClient({
      GET: () => ({
        data: [...toolPanel(1, 2), { id: "upload1", name: "Upload File", description: "Load data" }],
        response: { status: 200 },
      }),
    });
    const out = (await getToolPanel({}, ctxWith(client))) as ToolPanelOverview;
    expect(out.entries[1]).toEqual({ id: "upload1", name: "Upload File", type: "tool", description: "Load data" });
  });

  it("drills into one section and returns its slim tools", async () => {
    const out = (await getToolPanel({ sectionId: "section_2" }, ctxWith(serving(3, 4)))) as ToolPanelSection;
    expect(out.section_id).toBe("section_2");
    expect(out.section_name).toBe("Genomics Analysis / Section 2");
    expect(out.tools).toHaveLength(4);
    expect(Object.keys(out.tools[0]).sort()).toEqual(["description", "id", "name", "versions"]);
  });

  it("pages inside a section", async () => {
    const out = (await getToolPanel({ sectionId: "section_1", limit: 10, offset: 195 }, ctxWith(serving(2, 200)))) as ToolPanelSection;
    expect(out.tools).toHaveLength(5);
    expect(out.pagination).toMatchObject({ total: 200, hasNext: false, hasPrevious: true });
  });

  it("tells an agent how to find the valid ids when the section is unknown", async () => {
    await expect(getToolPanel({ sectionId: "nope" }, ctxWith(serving(3, 4)))).rejects.toThrow(
      /no tool panel section with id 'nope'.*get_tool_panel with no arguments/s,
    );
  });

  it("rejects a limit over the ceiling the Python tool sets, without calling Galaxy", async () => {
    const client = mockClient({ GET: () => { throw new Error("should not reach Galaxy"); } });
    await expect(getToolPanel({ limit: 501 }, ctxWith(client))).rejects.toThrow(/at most 500/);
  });

  it("handles a panel with nothing in it", async () => {
    const out = (await getToolPanel({}, ctxWith(serving(0, 0)))) as ToolPanelOverview;
    expect(out.entries).toEqual([]);
    expect(out.pagination).toMatchObject({ total: 0, hasNext: false });
  });

});

/**
 * The classification is the Python server's, copied: a node is a section when it
 * has an `elems` key, a tool when it has none and is not a ToolSectionLabel.
 * These cases are the ones where a rule of our own would answer differently, so
 * they are pinned against what the other surface does, not against what looks
 * tidier.
 */
describe("get_tool_panel node classification", () => {
  it("calls a ToolSection with no elems a tool, because the other surface does", async () => {
    const client = mockClient({
      GET: () => ({ data: [{ id: "s1", name: "Empty", model_class: "ToolSection" }], response: { status: 200 } }),
    });
    const out = (await getToolPanel({}, ctxWith(client))) as ToolPanelOverview;
    expect(out.entries[0]).toEqual({ id: "s1", name: "Empty", type: "tool", description: "" });
    // And it is not a section to drill into, for the same reason: Python looks up
    // a section by the elems key, so this one is not found there either.
    await expect(getToolPanel({ sectionId: "s1" }, ctxWith(client))).rejects.toThrow(/no tool panel section/);
  });

  it("keeps a real tool whose id happens to end in _label", async () => {
    const client = mockClient({
      GET: () => ({
        data: [
          {
            id: "s1",
            name: "Section",
            model_class: "ToolSection",
            elems: [
              { id: "make_label", name: "Make Label", model_class: "Tool", description: "d", versions: ["1"] },
              { id: "divider_label", name: "Divider", model_class: "ToolSectionLabel" },
            ],
          },
        ],
        response: { status: 200 },
      }),
    });
    const out = (await getToolPanel({ sectionId: "s1" }, ctxWith(client))) as ToolPanelSection;
    expect(out.tools.map((t) => t.id)).toEqual(["make_label"]);
  });

  it("keeps a classless node whose id ends in _label, because the other surface does", async () => {
    // No fallback on the id: Python has none, and a tool dropped on one surface
    // and kept on the other is the divergence this whole exercise is closing.
    const client = mockClient({
      GET: () => ({
        data: [{ id: "s1", name: "Section", elems: [{ id: "old_label" }, { id: "real_tool" }] }],
        response: { status: 200 },
      }),
    });
    const out = (await getToolPanel({ sectionId: "s1" }, ctxWith(client))) as ToolPanelSection;
    expect(out.tools.map((t) => t.id)).toEqual(["old_label", "real_tool"]);
  });
});
