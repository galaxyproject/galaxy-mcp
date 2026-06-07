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
  | "iwc";

/** The parsed input object derived from an op's raw Zod shape. */
export type InputOf<Shape extends ZodRawShape> = z.infer<ZodObject<Shape>>;

export interface Pagination {
  total?: number;
  offset?: number;
  limit?: number;
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
  readonly minGalaxyVersion?: string;
  /** Read-only by default. Write/mutating ops set this false (drives MCP annotations). */
  readonly readOnly?: boolean;
  run(input: InputOf<Shape>, ctx: GalaxyContext): Promise<O>;
  project?(output: O, input: InputOf<Shape>): { message?: string; pagination?: Pagination };
}

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
