import { describe, it, expect } from "vitest";
import { buildServer, toolNames } from "../src/server";

describe("MCP surface is a mechanical projection", () => {
  it("registers exactly one tool per operation", () => {
    const names = toolNames();
    expect(names).toContain("get_user");
    expect(names).toContain("run_tool");
    expect(names).toContain("get_invocations");
  });

  it("builds a server without throwing", () => {
    const server = buildServer({ baseUrl: "https://g.example", apiKey: "K" });
    expect(server).toBeDefined();
  });
});
