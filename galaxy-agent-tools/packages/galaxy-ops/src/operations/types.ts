import type { z, ZodRawShape, ZodObject } from "zod";
import type { GalaxyContext } from "../context";
import type { GalaxyErrorKind } from "../errors";

export type OperationDomain =
  | "connection"
  | "histories"
  | "datasets"
  | "collections"
  | "jobs"
  | "tools"
  | "userTools"
  | "workflows"
  | "invocations"
  | "iwc"
  | "pages";

/** The parsed input object derived from an op's raw Zod shape. */
export type InputOf<Shape extends ZodRawShape> = z.infer<ZodObject<Shape>>;

/**
 * Where a page sits, AS THE WIRE CARRIES IT.
 *
 * The Python server's PaginationInfo, key for key: snake_case names, every field
 * present, and null rather than absent where it has nothing to say. It is the
 * contract both surfaces answer with, so it is written the way that server writes
 * it rather than the way TypeScript would have. The library's own camelCase
 * `PaginationInfo` inside `Paged<T>` is a different thing and has not moved --
 * see pagination.ts.
 *
 * `trimmed_for_size` is deliberately missing. The other server has no such field
 * and folds the fact into helper_text ("This page was cut short to fit the output
 * budget..."), so an agent reads one sentence rather than checking a flag one
 * surface has.
 */
export interface Pagination {
  /** How many items exist in total, across every page. */
  total_items: number;
  /** How many items this page actually carries. */
  returned_items: number;
  limit: number;
  offset: number;
  has_next: boolean;
  has_previous: boolean;
  next_offset: number | null;
  previous_offset: number | null;
  /** One sentence telling an agent where it is and how to get the next page. */
  helper_text: string | null;
}

/**
 * An operation: identity, doc string, a Zod RAW SHAPE input (what MCP's
 * registerTool wants -- Record<string, ZodType>, NOT z.object(...)), and a run
 * that returns plain typed data or throws a typed error.
 */
export interface Operation<Shape extends ZodRawShape, O> {
  readonly name: string; // parity with AgentOperationsManager, e.g. "run_tool"
  readonly domain: OperationDomain;
  readonly summary: string; // reused verbatim as the MCP tool description
  readonly input: Shape; // raw shape -> MCP inputSchema directly
  /**
   * What the op needs from the server, as `{ galaxy: ">=26.1" }`. The registry refuses the
   * op before it runs against anything older, and both surfaces say so in their own words
   * for the op's description.
   */
  readonly requires?: { galaxy: string };
  /** Read-only by default. Write/mutating ops set this false (drives MCP annotations). */
  readonly readOnly?: boolean;
  /** Destructive (delete/cancel) ops set this true (drives MCP destructiveHint). */
  readonly destructive?: boolean;
  run(input: InputOf<Shape>, ctx: GalaxyContext): Promise<O>;
  /**
   * How to cut this op's page down, for the ops the Python server budgets.
   *
   * Set it and the surface measures the serialised result and trims until it fits
   * the output budget; leave it off and the result goes out whatever size it is,
   * which is what the two ops Python does not budget do.
   */
  budget?: { rows(data: O): number; shrink(data: O, keep: number): O };
  /**
   * How this op's output becomes the envelope a surface emits.
   *
   * Explicit and per op, because the shape the wire carries is not derivable from
   * the shape `run` returns: a paged op's `data` is the bare page with its
   * pagination lifted beside it, get_tool_panel keeps its own object, and a
   * ranking has no pagination at all. Leave `data` out and the op's own output
   * goes out unchanged, which is what every op that returns one record does.
   *
   * Called before the output budget is measured, and again on every trimming
   * pass, so what is weighed is the bytes the surface actually sends.
   */
  project?(output: O, input: InputOf<Shape>): Projection;
}

/** What `project` contributes to the envelope. */
export interface Projection {
  /** What the surface emits as `data`. Left out, the op's own output goes out. */
  data?: unknown;
  message?: string;
  /** Rows in this answer, for the tools that report one. Null where there is no count. */
  count?: number | null;
  pagination?: Pagination | null;
}

/** Heterogeneous registry element. */
export type AnyOperation = Operation<ZodRawShape, unknown>;

/**
 * The surface envelope (MCP/CLI projection of run()).
 *
 * The Python server's GalaxyResult: data, success, message, count, pagination.
 * `count` and `pagination` are always sent on a success, null where the tool has
 * none, because that is what a pydantic model serialises to and an agent should
 * not have to tell "this tool does not count" from "this surface forgot to say".
 *
 * `errorKind` is this surface's own, on the failure path only; the two servers
 * still fail differently and aligning that is its own piece of work.
 */
export interface GalaxyResult<T> {
  data: T;
  success: boolean;
  message?: string;
  count?: number | null;
  pagination?: Pagination | null;
  errorKind?: GalaxyErrorKind;
}
