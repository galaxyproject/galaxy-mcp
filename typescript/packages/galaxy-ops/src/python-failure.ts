/**
 * The sentence the Python MCP server says when a tool fails.
 *
 * That server's failure path is three layers, and all three are visible in what a client
 * reads. A tool catches whatever its HTTP client raised, `format_error(action, error,
 * context)` builds `"<Action> failed: <the exception's own text>"` plus a hint for the
 * statuses it recognises plus a rendered context dict, and FastMCP turns the raised
 * ValueError into an MCP error result whose text is `Error calling tool '<name>': ` and
 * then that sentence. Nothing in the middle layer is this project's prose: the exception's
 * text belongs to bioblend or to requests, and it is quoted verbatim.
 *
 * So a faithful port has to write those libraries' words too. There are three shapes, and
 * which one a caller sees depends on how the tool made the request rather than on what
 * went wrong -- see `PyRequestShape`. The facts each shape needs (status, URL, the reply's
 * bytes) travel with the error from the request that failed; the shape and the action come
 * from the operation, because they are facts about the tool and not about the reply.
 *
 * Measured against fastmcp 3.4.2, bioblend 1.9.0, requests 2.x and CPython 3.12, and every
 * shape here is pinned by a golden case under
 * python/tests/testdata/envelopes.
 */
import { pyStr } from "./python-values";

/**
 * Which client library worded the failure, and therefore how it reads.
 *
 * * `bioblend-get` -- a bioblend client GET (`gi.histories.show_history`, `gi.tools.show_tool`,
 *   ...). bioblend retries a GET, so the text carries the attempt counter, and it quotes the
 *   body twice: once as a Python `bytes` repr of the raw reply and once as decoded text.
 * * `bioblend-write` -- a bioblend POST, PUT, PATCH or DELETE, which says only the status and
 *   the body text.
 * * `raise-for-status` -- a request made with requests directly (`gi.make_get_request` and then
 *   `response.raise_for_status()`, or `requests.get`), which reports requests' own HTTPError:
 *   the status, the reason phrase, and the URL it could not read. This is the one shape whose
 *   text names the URL, so the two surfaces have to have asked the same question of Galaxy for
 *   it to read the same.
 */
export type PyRequestShape = "bioblend-get" | "bioblend-write" | "raise-for-status";

/** What a failed request leaves behind for the sentence to be built from. */
export interface HttpFailureFacts {
  /** The status Galaxy answered with, or null when the request never completed. */
  status: number | null;
  /** The method, so an operation that talks to two routes can tell which one failed. */
  method: string;
  /** The absolute URL as the request carried it, query string included. */
  url: string;
  /** The reply body as bytes-turned-text, exactly as it arrived. */
  bodyText: string;
  /** The status line's reason phrase, when the runtime hands one over. */
  reason?: string;
  /** What a runtime-level failure (no reply at all) said for itself. */
  transportMessage?: string;
}

/**
 * One hint per status, and the order matters: `format_error` falls back to searching the
 * exception's text for a status it recognises, first match wins, and a dict preserves
 * insertion order.
 */
export const PY_STATUS_HINTS: ReadonlyMap<number, string> = new Map([
  [401, "Authentication failed - check your API key"],
  [403, "Permission denied - check your account permissions"],
  [404, "Resource not found - check IDs and URLs"],
  [500, "Server error - try again later or contact admin"],
]);

/**
 * The reason phrases requests would have seen, for the runtimes that hand us none.
 *
 * requests reports whatever the status line said, and `fetch` exposes that as `statusText`
 * -- so the server's own phrase is used when there is one. There often is not: a `Response`
 * built in a test carries an empty `statusText`, and HTTP/2 has no reason phrase at all.
 * This is CPython's `http.client.responses` table, which is what the Python side's own test
 * double answers with, so the fallback agrees with the fixtures.
 *
 * It is a fallback and not the truth: the table travels with the Python version. 422 reads
 * "Unprocessable Entity" through 3.12 and "Unprocessable Content" from 3.13, so a sentence
 * quoting that particular status is not portable between interpreters and no case pins one.
 */
const PY_REASONS: Readonly<Record<number, string>> = {
  100: "Continue",
  101: "Switching Protocols",
  102: "Processing",
  103: "Early Hints",
  200: "OK",
  201: "Created",
  202: "Accepted",
  203: "Non-Authoritative Information",
  204: "No Content",
  205: "Reset Content",
  206: "Partial Content",
  207: "Multi-Status",
  208: "Already Reported",
  226: "IM Used",
  300: "Multiple Choices",
  301: "Moved Permanently",
  302: "Found",
  303: "See Other",
  304: "Not Modified",
  305: "Use Proxy",
  307: "Temporary Redirect",
  308: "Permanent Redirect",
  400: "Bad Request",
  401: "Unauthorized",
  402: "Payment Required",
  403: "Forbidden",
  404: "Not Found",
  405: "Method Not Allowed",
  406: "Not Acceptable",
  407: "Proxy Authentication Required",
  408: "Request Timeout",
  409: "Conflict",
  410: "Gone",
  411: "Length Required",
  412: "Precondition Failed",
  413: "Content Too Large",
  414: "URI Too Long",
  415: "Unsupported Media Type",
  416: "Range Not Satisfiable",
  417: "Expectation Failed",
  418: "I'm a Teapot",
  421: "Misdirected Request",
  422: "Unprocessable Entity",
  423: "Locked",
  424: "Failed Dependency",
  425: "Too Early",
  426: "Upgrade Required",
  428: "Precondition Required",
  429: "Too Many Requests",
  431: "Request Header Fields Too Large",
  451: "Unavailable For Legal Reasons",
  500: "Internal Server Error",
  501: "Not Implemented",
  502: "Bad Gateway",
  503: "Service Unavailable",
  504: "Gateway Timeout",
  505: "HTTP Version Not Supported",
  506: "Variant Also Negotiates",
  507: "Insufficient Storage",
  508: "Loop Detected",
  510: "Not Extended",
  511: "Network Authentication Required",
};

const hex = (code: number, width: number): string => code.toString(16).padStart(width, "0");

/**
 * Python's `repr()` of a `bytes`, which is how bioblend quotes a reply body it could not
 * use (`f"GET: error {status}: {r.content!r}"`).
 *
 * CPython's `bytes_repr`, condition for condition: the quote is `'` unless the bytes contain
 * one and no `"`; the chosen quote and `\` are escaped; tab, newline and carriage return get
 * their short escapes; every other byte outside the printable ASCII range is `\xhh` in
 * lowercase hex. The input here is text, so it is encoded as UTF-8 first -- which is what
 * Galaxy sends and what requests kept.
 */
export function pyBytesRepr(text: string): string {
  const bytes = new TextEncoder().encode(text);
  const quote = text.includes("'") && !text.includes('"') ? '"' : "'";
  let out = `b${quote}`;
  for (const byte of bytes) {
    const ch = String.fromCharCode(byte);
    if (ch === quote || ch === "\\") out += `\\${ch}`;
    else if (byte === 0x09) out += "\\t";
    else if (byte === 0x0a) out += "\\n";
    else if (byte === 0x0d) out += "\\r";
    else if (byte >= 0x20 && byte < 0x7f) out += ch;
    else out += `\\x${hex(byte, 2)}`;
  }
  return out + quote;
}

/**
 * The exception text the Python side's client library would have raised for this reply.
 *
 * `attemptsLeft` is bioblend's retry counter, and zero is not a guess: `max_get_attempts`
 * defaults to 1, so the one attempt is spent by the time it gives up and the text always
 * reads "0 attempts left". A server configured to retry would say more, and no tool here
 * configures one.
 */
export function pyLibraryText(shape: PyRequestShape, facts: HttpFailureFacts): string {
  const { status, bodyText, url } = facts;
  if (status === null) {
    // No reply to quote. bioblend substitutes an empty Response for a connection failure,
    // which leaves its GET text with an empty body and its counter intact; requests raises
    // its own error before any status exists, and that is the whole text.
    const said = facts.transportMessage ?? "";
    return shape === "bioblend-get" ? `${said}, 0 attempts left: ` : said;
  }
  if (shape === "bioblend-get") {
    return `GET: error ${status}: ${pyBytesRepr(bodyText)}, 0 attempts left: ${bodyText}`;
  }
  if (shape === "bioblend-write") {
    return `Unexpected HTTP status code: ${status}: ${bodyText}`;
  }
  const reason = facts.reason || PY_REASONS[status] || "";
  const kind = status >= 500 ? "Server Error" : "Client Error";
  return `${status} ${kind}: ${reason} for url: ${url}`;
}

/** `. Context: k=v, k2=v2`, or nothing at all for an empty dict. */
export function pyContextClause(context: Record<string, unknown>): string {
  const entries = Object.entries(context);
  if (entries.length === 0) return "";
  return `. Context: ${entries.map(([k, v]) => `${k}=${pyStr(v)}`).join(", ")}`;
}

/**
 * `format_error(action, error, context)`, hint rule included.
 *
 * The hint is chosen from the status the failure carries, never from its text -- the text is
 * requests' own whenever a request did not complete and requests quotes the URL in it, so an
 * identifier with "404" in it used to be reported as a missing resource. `status === null`
 * means the failure has no HTTP status at all and nothing is claimed about one.
 *
 * `statusIsKnown` is false for a failure that carries no status FIELD, which is not the same
 * as one that says there is no status: that is the single case Python leaves to a text
 * search, because there is nothing else to go on. It is what an error built out of a
 * message reaches -- the 200-with-an-error-body refusal, say.
 */
export function pyFormatError(
  action: string,
  errorText: string,
  status: number | null,
  context: Record<string, unknown> = {},
  { statusIsKnown = true }: { statusIsKnown?: boolean } = {},
): string {
  let msg = `${action} failed: ${errorText}`;
  if (status !== null) {
    const hint = PY_STATUS_HINTS.get(status);
    if (hint) msg += ` (${hint})`;
  } else if (!statusIsKnown) {
    for (const [code, hint] of PY_STATUS_HINTS) {
      if (errorText.includes(String(code))) {
        msg += ` (${hint})`;
        break;
      }
    }
  }
  return msg + pyContextClause(context);
}
