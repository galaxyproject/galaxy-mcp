import { describe, it, expect } from "vitest";
import { GalaxyValidationError } from "../../src/errors";
import { paginate, paginationInfo, validatePagination } from "../../src/operations/pagination";

describe("validatePagination", () => {
  it("rejects a limit over the ceiling, in the sentence the Python server uses", () => {
    expect(() => validatePagination(60, 0, { maxLimit: 35 })).toThrow(GalaxyValidationError);
    // Python's sentence, in Python's order: the count it got comes before the advice.
    expect(() => validatePagination(60, 0, { maxLimit: 35 })).toThrow(
      "limit must be at most 35 (got 60); request 35 or fewer and use offset to page through the rest",
    );
  });

  it("takes any positive limit for an op the Python server does not cap", () => {
    expect(() => validatePagination(100_000, 0)).not.toThrow();
    expect(() => validatePagination(0, 0)).toThrow(/at least 1/);
    expect(() => validatePagination(5, -1)).toThrow(/0 or greater/);
  });

  it("leaves the offset advice out for a listing that cannot page", () => {
    expect(() => validatePagination(26, 0, { maxLimit: 25, pageable: false })).toThrow(
      "limit must be at most 25 (got 26); request 25 or fewer",
    );
  });

  it("rejects a limit below one and a negative offset", () => {
    expect(() => validatePagination(0, 0, { maxLimit: 35 })).toThrow(/at least 1/);
    expect(() => validatePagination(5, -1, { maxLimit: 35 })).toThrow(/0 or greater/);
  });

  it("rejects a non-integer window rather than silently flooring it", () => {
    expect(() => validatePagination(2.5, 0, { maxLimit: 35 })).toThrow(/whole number/);
    expect(() => validatePagination(5, 1.5, { maxLimit: 35 })).toThrow(/0 or greater/);
  });

  it("accepts the cap itself", () => {
    expect(() => validatePagination(35, 0, { maxLimit: 35 })).not.toThrow();
  });
});

describe("how a rejected window reaches the caller", () => {
  it("is a validation error, not a connection failure", async () => {
    const { runWithEnvelope } = await import("../../src/operations/registry");
    const op = {
      name: "probe",
      domain: "tools" as const,
      summary: "",
      input: {},
      run: async () => {
        validatePagination(9999, 0, { maxLimit: 35 });
        return [];
      },
    };
    const result = await runWithEnvelope(op as never, {} as never, {} as never);
    // "connection" would tell an agent to back off and retry a call that can
    // never succeed, and would exit the CLI 69 for what the caller must fix.
    expect(result).toMatchObject({ success: false, errorKind: "validation" });
    expect(result.message).toMatch(/at most 35/);
  });
});

describe("paginate", () => {
  const items = Array.from({ length: 10 }, (_, n) => n);

  it("returns the first page and points at the next one", () => {
    const { items: page, pagination } = paginate(items, { limit: 4, offset: 0, noun: "things" });
    expect(page).toEqual([0, 1, 2, 3]);
    expect(pagination).toMatchObject({ total: 10, returned: 4, hasNext: true, hasPrevious: false, nextOffset: 4 });
    expect(pagination.previousOffset).toBeUndefined();
  });

  it("reports a short final page as the last one", () => {
    const { pagination } = paginate(items, { limit: 4, offset: 8, noun: "things" });
    expect(pagination).toMatchObject({ returned: 2, hasNext: false, hasPrevious: true, previousOffset: 4 });
    expect(pagination.nextOffset).toBeUndefined();
    expect(pagination.helperText).toContain("last page");
  });

  it("says so when the offset is past the end", () => {
    const { items: page, pagination } = paginate(items, { limit: 4, offset: 50, noun: "things" });
    expect(page).toEqual([]);
    expect(pagination.helperText).toBe("offset 50 is past the end of 10 things; use a smaller offset");
  });

  it("handles an empty collection without claiming an overrun", () => {
    const { pagination } = paginate([], { limit: 4, offset: 0, noun: "things" });
    expect(pagination).toMatchObject({ total: 0, returned: 0, hasNext: false, hasPrevious: false });
    expect(pagination.helperText).toBe("Showing 0 of 0 things (offset 0). This is the last page.");
  });

  it("walks every item exactly once across pages", () => {
    const seen: number[] = [];
    let offset = 0;
    for (;;) {
      const { items: page, pagination } = paginate(items, { limit: 3, offset, noun: "things" });
      seen.push(...page);
      if (pagination.nextOffset === undefined) break;
      offset = pagination.nextOffset;
    }
    expect(seen).toEqual(items);
  });
});

describe("paginationInfo", () => {
  it("never stalls a page walk when a server reports a total but returns nothing", () => {
    const info = paginationInfo({ total: 100, returned: 0, limit: 10, offset: 0, noun: "things" });
    expect(info.hasNext).toBe(true);
    expect(info.nextOffset).toBe(10);
  });

  it("does not let an offset past the end inflate the total", () => {
    const info = paginationInfo({ total: 10, returned: 0, limit: 5, offset: 40, noun: "things" });
    expect(info.total).toBe(10);
  });

  it("raises a stale total to the floor the items in hand prove", () => {
    const info = paginationInfo({ total: 3, returned: 5, limit: 5, offset: 10, noun: "things" });
    expect(info.total).toBe(15);
  });
});
