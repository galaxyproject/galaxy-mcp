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
    render({ data: { items: [], pagination: { total: 0 } }, success: true } as never, { format: "table", quiet: true });
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
          pagination: { total: 1 },
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
 * The output budget exists so what reaches the other end is readable, and this
 * surface prints indented JSON -- about a fifth larger than the single line the
 * MCP text block carries. Measuring the compact form and printing the indented
 * one put 58 KB on stdout for a page trimmed to just under 50 KB, so the
 * serializer is a parameter of the run and this checks the one the CLI passes.
 */
describe("a paged op through the CLI's own serializer", () => {
  const histories = Array.from({ length: 1000 }, (_, k) => ({
    id: String(k).padStart(16, "a"),
    name: `ゲノム解析パイプライン`.repeat(4) + ` (imported from GSE123456, replicate ${k})`,
  }));

  it("prints no more bytes than the budget it was trimmed to", async () => {
    const { listHistoryIdsOp, runWithEnvelope, DEFAULT_POLL, OUTPUT_BUDGET_BYTES } = await import(
      "@galaxyproject/galaxy-ops"
    );
    const { serializeForCli } = await import("../src/render");
    const client = { GET: async () => ({ data: histories, response: { status: 200 } }) } as never;
    const result = await runWithEnvelope(
      listHistoryIdsOp as never,
      { limit: 500, offset: 0 } as never,
      { client, poll: DEFAULT_POLL },
      serializeForCli as never,
    );
    const printed = serializeForCli(result as never);
    expect(new TextEncoder().encode(printed).length).toBeLessThanOrEqual(OUTPUT_BUDGET_BYTES);
    // And it really did have to cut something, or this proves nothing.
    expect((result.data as { items: unknown[] }).items.length).toBeLessThan(500);
  });
});
