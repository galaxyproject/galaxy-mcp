import type { components } from "./bindings";
import type { GalaxyContext } from "./context";
import { classifyHttp, ToolRequestRejectedError } from "./errors";
import { waitForJob, type JobDetail } from "./wait";

export type ToolRequestDetail = components["schemas"]["ToolRequestDetailedModel"];

/** Nested inputs ONLY -- data refs as {src,id}/{src,location,ext}, batches as {__class__:'Batch',values}. */
export type ToolInputs = Record<string, unknown>;

export interface ExecuteToolRequestArgs {
  toolId: string;
  historyId: string;
  inputs: ToolInputs;
  toolVersion?: string;
}

export interface ImplicitCollectionRef {
  src: "hdca";
  id: string;
  output_name: string;
}

export interface ToolRun {
  toolRequestId: string;
  jobs: JobDetail[];
  implicitCollections: ImplicitCollectionRef[];
  state: "ok";
  // v1 omits per-dataset `outputs` (derivable from jobs[].outputs); deferred to v2 alongside richer output shaping.
}

const TR = "/api/tool_requests/{id}" as const;
const TR_STATE = "/api/tool_requests/{id}/state" as const;

export async function executeToolRequest(
  args: ExecuteToolRequestArgs,
  ctx: GalaxyContext,
): Promise<ToolRun> {
  // Step 3: queue-and-return. 200 means QUEUED, not done.
  // Body is fully typed against JobRequest (no escape cast) so a future schema
  // drift in this -- the most important payload in the library -- fails the build.
  // send_email_notification is required by the generated type (Galaxy defaults it
  // server-side, but the binding marks it non-optional), so we send it explicitly.
  const submit = await ctx.client.POST("/api/jobs", {
    body: {
      tool_id: args.toolId,
      tool_version: args.toolVersion,
      history_id: args.historyId,
      inputs: args.inputs,
      strict: true,
      send_email_notification: false,
    },
  });
  if (submit.error || !submit.data) throw classifyHttp(submit.response.status, submit.error);
  const toolRequestId = submit.data.tool_request_id;

  // Step 4: poll /state until != 'new'. ToolRequestState has NO success state.
  const state = await pollRequestState(toolRequestId, ctx);
  if (state === "failed") {
    const detail = await fetchDetail(toolRequestId, ctx);
    const errMsg = extractErrMsg(detail.state_message);
    throw new ToolRequestRejectedError(args.toolId, errMsg, toolRequestId);
  }

  // Step 5: 'submitted' -> fetch jobs[] + implicit_collections[].
  const detail = await fetchDetail(toolRequestId, ctx);
  const jobRefs = (detail.jobs ?? []) as Array<{ src: "job"; id: string }>;
  const implicitCollections = (detail.implicit_collections ?? []) as ImplicitCollectionRef[];

  // Step 6: wait each job to terminal. Success is DERIVED FROM JOBS, never ToolRequest.state.
  const jobs: JobDetail[] = [];
  for (const ref of jobRefs) {
    jobs.push(await waitForJob(ref.id, ctx));
  }
  return { toolRequestId, jobs, implicitCollections, state: "ok" };
}

async function pollRequestState(id: string, ctx: GalaxyContext): Promise<string> {
  const start = Date.now();
  for (let attempt = 0; ; attempt++) {
    if (ctx.signal?.aborted) throw new Error("aborted");
    const { data, error, response } = await ctx.client.GET(TR_STATE, {
      params: { path: { id } },
    });
    if (error || data == null) throw classifyHttp(response.status, error);
    const state = String(data); // bare string enum
    if (state !== "new") return state;
    if (Date.now() - start > ctx.poll.timeoutMs) {
      throw new Error(`Timed out waiting for tool request ${id} to leave 'new'`);
    }
    await sleep(ctx.poll.intervalMs, ctx.signal);
  }
}

async function fetchDetail(id: string, ctx: GalaxyContext): Promise<ToolRequestDetail> {
  const { data, error, response } = await ctx.client.GET(TR, { params: { path: { id } } });
  if (error || !data) throw classifyHttp(response.status, error);
  return data;
}

/** state_message is a plain string in Galaxy 26.0 and an {err_msg} object in 26.1+. */
function extractErrMsg(stateMessage: unknown): string {
  if (typeof stateMessage === "string" && stateMessage) return stateMessage;
  if (stateMessage && typeof stateMessage === "object" && "err_msg" in stateMessage) {
    return String((stateMessage as { err_msg: unknown }).err_msg);
  }
  return "tool request failed to expand";
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((res, rej) => {
    if (signal?.aborted) return rej(new Error("aborted"));
    const t = setTimeout(res, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(t);
      rej(new Error("aborted"));
    }, { once: true });
  });
}
