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
