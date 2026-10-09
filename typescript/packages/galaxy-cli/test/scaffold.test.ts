import { describe, it, expect } from "vitest";
import { allOperations } from "@galaxyproject/galaxy-ops";

describe("galaxy-cli wiring", () => {
  it("can import the galaxy-ops registry", () => {
    expect(allOperations.length).toBeGreaterThan(10);
  });
});
