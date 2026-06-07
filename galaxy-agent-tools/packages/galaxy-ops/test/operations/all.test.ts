import { describe, it, expect } from "vitest";
import "../../src/operations/all";
import { allOperations } from "../../src/operations/registry";

describe("operations barrel", () => {
  it("registers the existing ops via a single import", () => {
    const names = allOperations.map((o) => o.name);
    expect(names).toContain("get_user");
    expect(names).toContain("run_tool");
    expect(names).toContain("get_invocations");
  });
});
