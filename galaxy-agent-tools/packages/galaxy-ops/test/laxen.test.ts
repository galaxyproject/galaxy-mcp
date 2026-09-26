import { describe, it, expect } from "vitest";
import { z } from "zod";
import { isPlainObject, laxenLikePydantic } from "../src/laxen";

/**
 * The decoding pydantic does for a tool call's arguments, unit by unit.
 *
 * Every row here was measured against the installed pydantic with
 * `TypeAdapter(int | None)` and `TypeAdapter(bool)`, in both `validate_python` and
 * `validate_json` -- they agree -- and the same rows are driven through both surfaces in
 * `galaxy-mcp/test/lax-arguments.test.ts` and `galaxy-cli/test/lax-arguments.test.ts`.
 */
const shape = {
  limit: z.number().int().optional(),
  version: z.number().int().nullish(),
  flag: z.boolean().optional(),
  ratio: z.number().optional(),
  label: z.string().optional(),
};

const decodes = (key: string, given: unknown): unknown => laxenLikePydantic(shape, { [key]: given })[key];

describe("an integer argument", () => {
  it.each([
    ["5", 5],
    [" 5 ", 5],
    ["+5", 5],
    ["-5", -5],
    ["05", 5],
    ["5.0", 5],
    ["5.00", 5],
    ["1_000", 1000],
    [5.0, 5],
    [-0.0, 0],
    [1e2, 100],
    [true, 1],
    [false, 0],
  ])("reads %j as %i", (given, want) => {
    expect(decodes("limit", given)).toBe(want);
    expect(decodes("version", given)).toBe(want);
  });

  it.each([["5.5"], [5.5], ["1e2"], ["0x10"], [""], ["abc"], ["٥"], ["1__0"], ["_1"], ["1_"], ["5."], ["."], [[1]], [{ a: 1 }], [Number.NaN], [Number.POSITIVE_INFINITY]])(
    "leaves %j for the schema to refuse",
    (given) => {
      expect(decodes("limit", given)).toBe(given);
    },
  );

  /**
   * Two quirks of pydantic's own parser that this does not copy, and neither did anything
   * before it: a leading zero makes it forgiving about what follows, so "0__5" is 5 and
   * "0-5" is -5 over there and a refusal here. Worth knowing about, not worth reproducing --
   * nothing sends them on purpose.
   */
  it.each([["0__5"], ["0-5"]])("refuses pydantic's leading-zero quirk %j", (given) => {
    expect(decodes("limit", given)).toBe(given);
  });

  it("agrees with pydantic on the ordinary leading-zero forms", () => {
    // "0_5" and "00_5" are 5 on both sides; it is the doubled underscore and the embedded
    // sign that only pydantic's parser lets through.
    expect(decodes("limit", "0_5")).toBe(5);
    expect(decodes("limit", "00_5")).toBe(5);
  });

  it("leaves an integer JavaScript cannot hold, rather than changing it", () => {
    // pydantic keeps 9007199254740993 exactly; Number() would hand back ...92.
    expect(decodes("limit", "9007199254740993")).toBe("9007199254740993");
  });

  it("leaves null and undefined for the schema to rule on", () => {
    expect(decodes("limit", null)).toBeNull();
    expect(laxenLikePydantic(shape, {})).toEqual({});
  });

  it("does not touch a field that is not an integer", () => {
    expect(decodes("ratio", "5")).toBe("5");
    expect(decodes("label", "5")).toBe("5");
  });
});

/**
 * The whitespace pydantic strips, code point by code point, because it is not the whitespace
 * `String.prototype.trim` strips. Measured with `TypeAdapter(int | None)`: everything in the
 * first list is trimmed there and everything in the second is refused.
 *
 * The two that disagree with `trim()` are the reason this is a table and not a call to it:
 * U+FEFF is stripped by JavaScript and refused by pydantic, and U+0085 is the other way
 * round.
 */
describe("the whitespace around an integer", () => {
  it.each([
    ["\t", "tab"],
    ["\n", "line feed"],
    ["\v", "vertical tab"],
    ["\f", "form feed"],
    ["\r", "carriage return"],
    [" ", "space"],
    ["", "next line -- trim() does NOT strip this one"],
    [" ", "no-break space"],
    [" ", "ogham space mark"],
    [" ", "en quad"],
    [" ", "figure space"],
    [" ", "hair space"],
    [" ", "line separator"],
    [" ", "paragraph separator"],
    [" ", "narrow no-break space"],
    [" ", "medium mathematical space"],
    ["　", "ideographic space"],
  ])("strips %j (%s)", (space) => {
    expect(decodes("limit", `${space}5${space}`)).toBe(5);
  });

  it.each([
    ["﻿", "zero width no-break space -- trim() strips this one, pydantic refuses it"],
    ["᠎", "mongolian vowel separator"],
    ["​", "zero width space"],
  ])("does not strip %j (%s)", (space) => {
    expect(decodes("limit", `${space}5${space}`)).toBe(`${space}5${space}`);
  });
});

describe("a boolean argument", () => {
  it.each([
    ["true", true],
    ["false", false],
    ["True", true],
    ["FALSE", false],
    ["t", true],
    ["f", false],
    ["yes", true],
    ["no", false],
    ["y", true],
    ["n", false],
    ["on", true],
    ["off", false],
    ["Off", false],
    ["1", true],
    ["0", false],
    [1, true],
    [0, false],
    [1.0, true],
    [0.0, false],
  ])("reads %j as %j", (given, want) => {
    expect(decodes("flag", given)).toBe(want);
  });

  it.each([[2], [-1], [""], [" true "], ["maybe"], [[true]], [{}]])(
    "leaves %j for the schema to refuse",
    (given) => {
      expect(decodes("flag", given)).toBe(given);
    },
  );
});

/**
 * One definition of "plain object", used by the decoder and by the surface that feeds it.
 * `typeof` is not it: everything in the second list answers "object" and none of them is a
 * set of arguments, and a transport that carries references rather than JSON text can
 * deliver any of them.
 */
describe("what counts as a plain object", () => {
  it.each([[{}], [{ limit: 5 }], [Object.create(null) as object]])("takes %j", (value) => {
    expect(isPlainObject(value)).toBe(true);
  });

  it.each([
    [[], "an array"],
    [new Date(0), "a Date"],
    [new Map(), "a Map"],
    [new Set(), "a Set"],
    // eslint-disable-next-line no-new-wrappers
    [new Number(5), "a boxed number"],
    // eslint-disable-next-line no-new-wrappers
    [new String("x"), "a boxed string"],
    [new (class Args {})(), "a class instance"],
    [null, "null"],
    [5, "a number"],
    ["x", "a string"],
  ])("refuses %s", (value) => {
    expect(isPlainObject(value)).toBe(false);
  });

  it("hands back anything that is not one, rather than spreading it", () => {
    const notArguments = new Date(0) as unknown as Record<string, unknown>;
    expect(laxenLikePydantic(shape, notArguments)).toBe(notArguments);
  });
});
