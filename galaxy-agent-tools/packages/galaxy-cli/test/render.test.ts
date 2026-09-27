import { describe, it, expect, vi } from "vitest";
import { render } from "../src/render";
import type { GalaxyResult } from "@galaxyproject/galaxy-ops";

const ok: GalaxyResult<unknown> = { data: [{ id: "h1", name: "alpha" }], success: true, message: "1 history" };

describe("render", () => {
  it("json mode prints the full envelope to stdout", () => {
    const out = vi.spyOn(console, "log").mockImplementation(() => {});
    render(ok, { format: "json", quiet: false });
    expect(out).toHaveBeenCalledWith(JSON.stringify(ok, null, 2));
    out.mockRestore();
  });
  it("table mode prints columns to stdout and the message to stderr", () => {
    const out = vi.spyOn(console, "log").mockImplementation(() => {});
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    render(ok, { format: "table", quiet: false });
    expect(out.mock.calls.flat().join("\n")).toMatch(/id.*name/);
    expect(err).toHaveBeenCalledWith("1 history");
    out.mockRestore(); err.mockRestore();
  });
  it("quiet suppresses the message", () => {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "log").mockImplementation(() => {});
    render(ok, { format: "table", quiet: true });
    expect(err).not.toHaveBeenCalled();
    vi.restoreAllMocks();
  });
});

/**
 * A paged list op's real envelope, built by running the op rather than by hand,
 * so this breaks if the ops layer changes its result shape again. There was no
 * CLI test over a list op before, which is how a renderer that printed
 * "items [100]" instead of a table went unnoticed.
 */
describe("render a paged list op", () => {
  const paged = async () => {
    const { listHistoryIdsOp, runWithEnvelope, DEFAULT_POLL } = await import("@galaxyproject/galaxy-ops");
    const histories = Array.from({ length: 3 }, (_, k) => ({ id: `h${k + 1}`, name: `history ${k + 1}` }));
    const client = { GET: async () => ({ data: histories, response: { status: 200 } }) } as never;
    return runWithEnvelope(listHistoryIdsOp as never, { limit: 2, offset: 0 } as never, {
      client,
      poll: DEFAULT_POLL,
    });
  };

  it("tables the rows rather than printing the page wrapper", async () => {
    const result = await paged();
    const out = vi.spyOn(console, "log").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
    render(result as never, { format: "table", quiet: true });
    const printed = out.mock.calls.flat().join("\n");
    expect(printed).toMatch(/id\s+name/);
    expect(printed).toContain("h1");
    expect(printed).not.toContain("items");
    expect(printed).not.toContain("pagination");
    vi.restoreAllMocks();
  });

  it("puts the pagination helper on stderr so an agent can page on", async () => {
    const result = await paged();
    vi.spyOn(console, "log").mockImplementation(() => {});
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    render(result as never, { format: "table", quiet: false });
    expect(err.mock.calls.flat().join("\n")).toMatch(/Showing 2 of 3 histories/);
    vi.restoreAllMocks();
  });

  it("says (empty) for a page with no rows", () => {
    const out = vi.spyOn(console, "log").mockImplementation(() => {});
    render({ data: [], success: true } as never, { format: "table", quiet: true });
    expect(out.mock.calls.flat().join("\n")).toBe("(empty)");
    vi.restoreAllMocks();
  });

  it("says (empty) for a history whose page of contents is empty", () => {
    const out = vi.spyOn(console, "log").mockImplementation(() => {});
    render({ data: { history_id: "h1", contents: [] }, success: true } as never, {
      format: "table",
      quiet: true,
    });
    expect(out.mock.calls.flat().join("\n")).toBe("(empty)");
    vi.restoreAllMocks();
  });

  it("still key-values an ordinary object that is not a page", () => {
    const out = vi.spyOn(console, "log").mockImplementation(() => {});
    render({ data: { id: "h1", name: "alpha" }, success: true } as never, { format: "table", quiet: true });
    const printed = out.mock.calls.flat().join("\n");
    expect(printed).toMatch(/id\s+h1/);
    expect(printed).toMatch(/name\s+alpha/);
    vi.restoreAllMocks();
  });

  it("tables the tools of a tool panel section, which uses a different key", () => {
    const out = vi.spyOn(console, "log").mockImplementation(() => {});
    render(
      {
        data: {
          section_id: "s1",
          section_name: "Mapping",
          tools: [{ id: "bwa", name: "BWA", description: "map reads", versions: ["1.0"] }],
        },
        success: true,
      } as never,
      { format: "table", quiet: true },
    );
    const printed = out.mock.calls.flat().join("\n");
    expect(printed).toMatch(/id\s+name/);
    expect(printed).toContain("bwa");
    vi.restoreAllMocks();
  });
});

/**
 * The budget is measured on the compact line, not on the indented printing.
 *
 * This surface prints JSON about a fifth larger than the single line the MCP
 * text block carries, and it used to measure what it printed -- so the same call
 * came back with fewer rows here than there, and a page became a property of who
 * was reading it. It measures what a model reads now: the same bytes, the same
 * rows, and stdout is simply allowed to be bigger than the budget.
 */
describe("a paged op through the CLI", () => {
  const histories = Array.from({ length: 1000 }, (_, k) => ({
    id: String(k).padStart(16, "a"),
    name: `ゲノム解析パイプライン`.repeat(4) + ` (imported from GSE123456, replicate ${k})`,
  }));

  it("cuts the page where the MCP surface cuts it, then prints it indented", async () => {
    const { listHistoryIdsOp, runWithEnvelope, DEFAULT_POLL, OUTPUT_BUDGET_BYTES } = await import(
      "@galaxyproject/galaxy-ops"
    );
    const { printJson } = await import("../src/render");
    const client = { GET: async () => ({ data: histories, response: { status: 200 } }) } as never;
    const ctx = { client, poll: DEFAULT_POLL };
    const input = { limit: 500, offset: 0 };

    // What the CLI does: no serializer of its own.
    const cli = await runWithEnvelope(listHistoryIdsOp as never, input as never, ctx as never);
    // What the MCP surface does, which is the same thing.
    const mcp = await runWithEnvelope(listHistoryIdsOp as never, input as never, ctx as never);
    expect((cli.data as unknown[]).length).toBe((mcp.data as unknown[]).length);
    // And it really did have to cut something, or this proves nothing.
    expect((cli.data as unknown[]).length).toBeLessThan(500);

    const encoder = new TextEncoder();
    expect(encoder.encode(JSON.stringify(cli)).length).toBeLessThanOrEqual(OUTPUT_BUDGET_BYTES);
    // The indentation is for a human and is allowed to push the printing past the
    // budget; what was measured is what a model would have read.
    expect(encoder.encode(printJson(cli as never)).length).toBeGreaterThan(OUTPUT_BUDGET_BYTES);
  });
});
