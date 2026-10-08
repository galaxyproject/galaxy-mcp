/**
 * The Python server's failure prose, at the branches no golden case reaches.
 *
 * Every shape here is pinned end to end by a case under
 * python/tests/testdata/envelopes, replayed through both surfaces. What that
 * cannot reach is the exotic input: a reply body carrying a quote, or bytes that are not
 * ASCII, or a failure with no status on it at all. Those are checked here, against values
 * read off the interpreter rather than recalled -- `repr(text.encode("utf-8"))` on CPython
 * 3.12 for the bytes, and `format_error`'s own branches for the rest.
 */
import { describe, it, expect } from "vitest";
import {
  pyBytesRepr,
  pyContextClause,
  pyFormatError,
  pyLibraryText,
  PY_STATUS_HINTS,
  type HttpFailureFacts,
} from "../src/python-failure";

const facts = (over: Partial<HttpFailureFacts> = {}): HttpFailureFacts => ({
  status: 404,
  method: "GET",
  url: "https://galaxy.example/api/pages/p1",
  bodyText: '{"err_msg": "nope"}',
  ...over,
});

describe("repr() of the bytes bioblend quotes a reply body with", () => {
  it("quotes with an apostrophe by default", () => {
    expect(pyBytesRepr('{"err_msg": "nope"}')).toBe('b\'{"err_msg": "nope"}\'');
  });

  it("switches to double quotes for a body carrying an apostrophe and no double quote", () => {
    expect(pyBytesRepr("it's not here")).toBe('b"it\'s not here"');
  });

  it("keeps the apostrophe quote when the body has a double quote too, and escapes", () => {
    expect(pyBytesRepr('he said "no"')).toBe("b'he said \"no\"'");
    expect(pyBytesRepr('both \' and "')).toBe("b'both \\' and \"'");
  });

  it("escapes every byte outside printable ASCII, one byte at a time", () => {
    // Not one escape per character: it is a bytes repr, so a two-byte code point is two.
    expect(pyBytesRepr("café … nope")).toBe("b'caf\\xc3\\xa9 \\xe2\\x80\\xa6 nope'");
    expect(pyBytesRepr("emoji 🧬")).toBe("b'emoji \\xf0\\x9f\\xa7\\xac'");
  });

  it("uses the short escapes for tab, newline and carriage return", () => {
    expect(pyBytesRepr("line\nbreak\ttab\r")).toBe("b'line\\nbreak\\ttab\\r'");
  });

  it("writes the other control bytes and DEL as \\xhh", () => {
    expect(pyBytesRepr("\x00\x1f\x7f")).toBe("b'\\x00\\x1f\\x7f'");
  });

  it("escapes a backslash", () => {
    expect(pyBytesRepr("a\\b")).toBe("b'a\\\\b'");
  });
});

describe("the three shapes a client library reports a refused request in", () => {
  it("quotes the body twice for a bioblend GET, with the attempt counter", () => {
    expect(pyLibraryText("bioblend-get", facts({ status: 500 }))).toBe(
      'GET: error 500: b\'{"err_msg": "nope"}\', 0 attempts left: {"err_msg": "nope"}',
    );
  });

  it("says only the status and the body for a bioblend write", () => {
    expect(pyLibraryText("bioblend-write", facts({ status: 400 }))).toBe(
      'Unexpected HTTP status code: 400: {"err_msg": "nope"}',
    );
  });

  it("names the URL for a request checked with raise_for_status", () => {
    expect(pyLibraryText("raise-for-status", facts())).toBe(
      "404 Client Error: Not Found for url: https://galaxy.example/api/pages/p1",
    );
  });

  it("says Server Error from 500 up, and takes the reason the runtime hands over", () => {
    expect(pyLibraryText("raise-for-status", facts({ status: 503 }))).toBe(
      "503 Server Error: Service Unavailable for url: https://galaxy.example/api/pages/p1",
    );
    expect(pyLibraryText("raise-for-status", facts({ status: 418, reason: "Teapot" }))).toBe(
      "418 Client Error: Teapot for url: https://galaxy.example/api/pages/p1",
    );
  });

  it("falls back to the status table when the runtime hands over an empty reason", () => {
    // Which is what a Response built in a test carries, and what HTTP/2 has.
    expect(pyLibraryText("raise-for-status", facts({ reason: "" }))).toContain("Not Found");
  });

  it("reports a request that never got a reply as what the runtime said", () => {
    const dead = facts({ status: null, bodyText: "", transportMessage: "fetch failed" });
    // bioblend substitutes an empty Response for a connection failure, which leaves its GET
    // text with an empty body and its counter intact; requests raises before a status exists.
    expect(pyLibraryText("bioblend-get", dead)).toBe("fetch failed, 0 attempts left: ");
    expect(pyLibraryText("raise-for-status", dead)).toBe("fetch failed");
    expect(pyLibraryText("bioblend-write", dead)).toBe("fetch failed");
  });
});

describe("format_error", () => {
  it("adds the hint for the four statuses it knows, and nothing for the rest", () => {
    expect(pyFormatError("Get page", "boom", 401)).toBe(
      "Get page failed: boom (Authentication failed - check your API key)",
    );
    expect(pyFormatError("Get page", "boom", 403)).toContain("(Permission denied");
    expect(pyFormatError("Get page", "boom", 404)).toContain("(Resource not found");
    expect(pyFormatError("Get page", "boom", 500)).toContain("(Server error");
    expect(pyFormatError("Get page", "boom", 400)).toBe("Get page failed: boom");
    expect(pyFormatError("Get page", "boom", 422)).toBe("Get page failed: boom");
  });

  it("says nothing about a status that is not there", () => {
    // A failure with no HTTP status has nothing true to say about one, and 404 in the text
    // is not evidence: requests quotes the URL it could not reach, so an id with 404 in it
    // used to be reported as a missing resource.
    expect(pyFormatError("Get page", "connecting to id404 failed", null)).toBe(
      "Get page failed: connecting to id404 failed",
    );
  });

  it("searches the text only for a failure that carries no status field at all", () => {
    // Which is the one case left to a text search over there, because there is nothing else
    // to go on -- an error built out of a message, as the 200-with-an-error-body refusal is.
    expect(
      pyFormatError("Get workflow invocations", "err 404 something", null, {}, {
        statusIsKnown: false,
      }),
    ).toBe("Get workflow invocations failed: err 404 something (Resource not found - check IDs and URLs)");
  });

  it("takes the first hint in the table's order, not the first in the text", () => {
    expect(
      pyFormatError("X", "500 came after 403", null, {}, { statusIsKnown: false }),
    ).toContain("(Permission denied - check your account permissions)");
    expect([...PY_STATUS_HINTS.keys()]).toEqual([401, 403, 404, 500]);
  });

  it("renders the context the way an f-string renders each value", () => {
    expect(
      pyFormatError("Run tool", "boom", 400, {
        history_id: "h1",
        tool_id: null,
        inputs: { a: 1, b: "x" },
        published: false,
      }),
    ).toBe(
      "Run tool failed: boom. Context: history_id=h1, tool_id=None, " +
        "inputs={'a': 1, 'b': 'x'}, published=False",
    );
  });

  it("adds no context clause at all for an empty dict", () => {
    expect(pyContextClause({})).toBe("");
    expect(pyFormatError("Update history", "boom", 400)).toBe("Update history failed: boom");
  });
});
