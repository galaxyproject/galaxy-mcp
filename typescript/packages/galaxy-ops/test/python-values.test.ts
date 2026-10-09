import { describe, it, expect } from "vitest";
import { pyGet, pyStr } from "../src/python-values";

describe("pyGet is dict.get, not ??", () => {
  it("defaults an absent key only", () => {
    expect(pyGet({}, "name", "Unnamed")).toBe("Unnamed");
    expect(pyGet({ name: null }, "name", "Unnamed")).toBe(null);
    expect(pyGet({ name: "" }, "name", "Unnamed")).toBe("");
    expect(pyGet({ name: false }, "name", "Unnamed")).toBe(false);
    expect(pyGet({ name: "Reads" }, "name", "Unnamed")).toBe("Reads");
  });

  it("reads the record's own keys and not the prototype's", () => {
    // `"toString" in record` is true for every object, so an own-key test is the only
    // one that answers the question dict.get answers.
    expect(pyGet({}, "toString", "absent")).toBe("absent");
    expect(pyGet({}, "constructor", "absent")).toBe("absent");
  });
});

describe("pyStr is str(), which is what an f-string renders", () => {
  it("leaves a string alone", () => {
    expect(pyStr("Reads QC")).toBe("Reads QC");
    // Not repr: no quotes are added, because the sentence supplies its own.
    expect(pyStr("it's")).toBe("it's");
  });

  it("renders the absent value the way Python names it", () => {
    expect(pyStr(null)).toBe("None");
    expect(pyStr(undefined)).toBe("None");
  });

  it("capitalises a boolean, as Python spells one", () => {
    expect(pyStr(true)).toBe("True");
    expect(pyStr(false)).toBe("False");
  });

  it("renders a whole number as an integer", () => {
    expect(pyStr(0)).toBe("0");
    expect(pyStr(42)).toBe("42");
  });

  it("renders a container through repr, as str() of one does", () => {
    expect(pyStr(["a", null, true])).toBe("['a', None, True]");
    expect(pyStr({ name: "a", tags: [] })).toBe("{'name': 'a', 'tags': []}");
  });
});
