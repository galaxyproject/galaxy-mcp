/**
 * The stated order for a workflow's steps, and why it is stated.
 *
 * The rule: numeric key ascending for keys that are non-negative integers written
 * canonically (no leading zeros, no sign), then the remaining keys in the order
 * they arrived. It exists because a JSON object has no order the two servers
 * agree on -- Python keeps arrival order, this runtime hoists integer-like keys --
 * so neither inherits one.
 */
import { describe, it, expect } from "vitest";
import { canonicalStepIndex, stepsInOrder } from "../src/workflow-steps";

const keys = (steps: Record<string, unknown>) => stepsInOrder(steps).map(([k]) => k);

describe("stepsInOrder", () => {
  it("visits canonical numeric keys ascending, then the rest as they arrived", () => {
    expect(keys({ "2": 0, "10": 0, "1": 0, x: 0 })).toEqual(["1", "2", "10", "x"]);
  });

  it("keeps the non-index keys in arrival order rather than sorting them", () => {
    expect(keys({ zeta: 0, "1": 0, alpha: 0, "0": 0 })).toEqual(["0", "1", "zeta", "alpha"]);
  });

  /**
   * The case that proves nothing here leans on the runtime's own order. A key past
   * the array-index range is an ordinary string key to this runtime, so
   * Object.entries hands those two back in arrival order -- which is descending.
   */
  it("sorts numeric keys past the array-index range, which the runtime does not", () => {
    const steps = { "9999999999": 0, "4294967296": 0, "5": 0 };
    expect(Object.keys(steps)).toEqual(["5", "9999999999", "4294967296"]);
    expect(keys(steps)).toEqual(["5", "4294967296", "9999999999"]);
  });

  it("compares numeric keys by value, not as text", () => {
    expect(keys({ "9": 0, "100": 0, "20": 0 })).toEqual(["9", "20", "100"]);
  });

  it("is an empty list for an object with no steps", () => {
    expect(keys({})).toEqual([]);
  });
});

describe("canonicalStepIndex", () => {
  it("takes a canonical non-negative integer and nothing else", () => {
    expect(canonicalStepIndex("0")).toBe(0);
    expect(canonicalStepIndex("42")).toBe(42);
    for (const key of ["007", "+1", "-1", "1e2", "0x10", "1.0", " 1 ", "", "abc123"]) {
      expect(canonicalStepIndex(key), key).toBeNull();
    }
  });
});
