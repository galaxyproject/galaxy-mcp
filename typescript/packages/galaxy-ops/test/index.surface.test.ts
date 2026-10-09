import { describe, it, expect } from "vitest";
import { isJobTerminal, isJobSuccess } from "../src/index";

describe("public terminal-state predicates", () => {
  it("are re-exported from the package entry", () => {
    expect(isJobTerminal("ok")).toBe(true);
    expect(isJobTerminal("running")).toBe(false);
    expect(isJobSuccess("ok")).toBe(true);
    expect(isJobSuccess("error")).toBe(false);
  });
});
