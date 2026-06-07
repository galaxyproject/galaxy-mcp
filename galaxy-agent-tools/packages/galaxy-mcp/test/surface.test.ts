import { describe, it, expect } from "vitest";
import { buildServer, toolNames, toolAnnotations } from "../src/server";

describe("MCP surface is a mechanical projection", () => {
  it("registers one tool per registered op", () => {
    const names = toolNames();
    expect(names).toContain("get_user");
    expect(names).toContain("get_histories");
    expect(names).toContain("create_history");
    expect(names).toContain("get_tool_details");
  });

  it("marks read ops readOnly and writes not", () => {
    const ann = toolAnnotations();
    expect(ann.get_histories?.readOnlyHint).toBe(true);
    expect(ann.create_history?.readOnlyHint).toBe(false);
    expect(ann.create_history?.destructiveHint).toBe(false);
  });

  it("builds a server without throwing", () => {
    expect(buildServer({ baseUrl: "https://g.example", apiKey: "K" })).toBeDefined();
  });
});
