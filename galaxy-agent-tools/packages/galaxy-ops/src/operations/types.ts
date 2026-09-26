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

/** The input side, not the parsed side: nothing validates before run(), so a default is run()'s. */
export type InputOf<Shape extends ZodRawShape> = z.input<ZodObject<Shape>>;

export interface Pagination {
  total?: number;
  offset?: number;
  limit?: number;
  /** Items on this page. Absent on ops that do not window. */
  returned?: number;
  hasNext?: boolean;
  hasPrevious?: boolean;
  nextOffset?: number;
  previousOffset?: number;
  /** One sentence telling an agent where it is and how to get the next page. */
  helperText?: string;
  /** The page was cut to fit the output budget, not because there is nothing more. */
  trimmedForSize?: boolean;
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
  /** What the result is; declared only where this package owns the shape rather than Galaxy. */
  readonly result?: ResultShape;
  run(input: InputOf<Shape>, ctx: GalaxyContext): Promise<O>;
  /**
   * How to cut this op's page down, for the ops the Python server budgets.
   *
   * Set it and the surface measures the serialised result and trims until it fits
   * the output budget; leave it off and the result goes out whatever size it is,
   * which is what the two ops Python does not budget do.
   */
  budget?: { rows(data: O): number; shrink(data: O, keep: number): O };
  project?(output: O, input: InputOf<Shape>): { message?: string; pagination?: Pagination };
}

/**
 * A result is a sequence of rows or an object with named keys, and either may page.
 *
 * `fields` is left out when the keys vary by branch: the kind is still a fact then, the list is
 * not. `paginated` says the envelope carries the page window beside the data, which a test holds
 * to what `project` really emits. The Python surface states the same three things, so the parity
 * report compares them.
 */
export type ResultShape = { paginated?: boolean } & (
  | { kind: "object"; fields?: readonly string[] }
  | { kind: "list" }
);

/** Heterogeneous registry element. */
export type AnyOperation = Operation<ZodRawShape, unknown>;

/** The surface envelope (MCP/CLI projection of run()). */
export interface GalaxyResult<T> {
  data: T;
  success: boolean;
  message?: string;
  pagination?: Pagination;
  errorKind?: GalaxyErrorKind;
}
