import { describe, it, expect } from "vitest";
import "../../src/operations/all";
import { allOperations } from "../../src/operations/registry";

const PAGE_OPS = [
  "list_pages",
  "get_page",
  "create_page",
  "update_page",
  "list_page_revisions",
  "get_page_revision",
  "revert_page_revision",
];

describe("pages ops registration", () => {
  it("every page op is reachable through the barrel import", () => {
    const names = allOperations.map((o) => o.name);
    for (const name of PAGE_OPS) expect(names).toContain(name);
  });

  it("every page op sits in the pages domain", () => {
    const pages = allOperations.filter((o) => PAGE_OPS.includes(o.name));
    expect(pages).toHaveLength(PAGE_OPS.length);
    expect([...new Set(pages.map((o) => o.domain))]).toEqual(["pages"]);
  });

  it("the three writes are the only page ops the MCP surface will not annotate read-only", () => {
    const pages = allOperations.filter((o) => PAGE_OPS.includes(o.name));
    const writes = pages.filter((o) => (o.readOnly ?? true) === false).map((o) => o.name).sort();
    expect(writes).toEqual(["create_page", "revert_page_revision", "update_page"]);
  });

  it("no page op is destructive -- none of them delete anything", () => {
    const pages = allOperations.filter((o) => PAGE_OPS.includes(o.name));
    expect(pages.filter((o) => (o.destructive ?? false) === true)).toEqual([]);
  });
});
