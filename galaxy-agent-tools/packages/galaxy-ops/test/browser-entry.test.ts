import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
// Only the browser entry, and nothing else: the registry is one module-level array, so a
// file that also imported the full entry would be asserting against a registry both filled.
import { allOperations as browserOperations } from "../src/index.browser";

/**
 * The browser entry is a promise about what a bundle can do, so it is checked against the
 * source rather than trusted. A registry that lists an op a browser cannot run would be
 * discovered by a caller at the moment it failed, which is too late to be useful.
 */
const OPS_DIR = join(__dirname, "..", "src", "operations");

/** Op modules whose implementation reaches for something only Node has. */
function needsNode(): string[] {
  return readdirSync(OPS_DIR)
    .filter((f) => f.endsWith(".ts"))
    .filter((f) => /from "node:/.test(readFileSync(join(OPS_DIR, f), "utf8")))
    .map((f) => f.replace(/\.ts$/, ""))
    .sort();
}

describe("the browser entry", () => {
  it("registers no op that needs a filesystem", () => {
    const names = browserOperations.map((o) => o.name);
    expect(names.length).toBeGreaterThan(0);
    expect(names).not.toContain("upload_file");
    expect(names).not.toContain("download_dataset");
  });

  it("registers every op that does not need one", () => {
    // Counted from the modules that call register(), so a new op is included without anyone
    // remembering to update a number here.
    const defined = readdirSync(OPS_DIR)
      .filter((f) => f.endsWith(".ts"))
      // The call, at top level -- registry.ts defines register() and must not count itself.
      .filter((f) => /^register\(/m.test(readFileSync(join(OPS_DIR, f), "utf8"))).length;
    expect(browserOperations.length).toBe(defined - needsNode().length);
  });

  it("leaves out exactly the op modules that import a node builtin", () => {
    // If an op grows a node: import, it has to leave all-browser in the same change.
    expect(needsNode()).toEqual(["download-dataset", "upload-file"]);
  });

  it("pulls no node builtin into its own module graph", () => {
    const listed = readFileSync(join(OPS_DIR, "all-browser.ts"), "utf8");
    for (const mod of needsNode()) {
      expect(listed).not.toContain(`"./${mod}"`);
    }
  });
});
