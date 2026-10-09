import type { GalaxyClient } from "../../src/client";

type Resp = { data?: unknown; error?: unknown; response?: { status: number; headers?: Headers } };
type Handler = (path: string, init?: any) => Resp | Promise<Resp>;

/** Hand-rolled GalaxyClient stand-in. Ops call .GET/.POST/.PUT/.DELETE. */
export function mockClient(routes: {
  GET?: Handler;
  POST?: Handler;
  PUT?: Handler;
  DELETE?: Handler;
}): GalaxyClient {
  const fail = (m: string): Resp => ({ error: m, response: { status: 500 } });
  // An op may read a response header -- list_pages takes its total from one -- and a
  // reply without them is a reply openapi-fetch would never produce, so fill in an
  // empty set rather than letting a test crash on a missing `headers`.
  const withHeaders = async (r: Resp | Promise<Resp>): Promise<Resp> => {
    const resp = await r;
    const status = resp.response?.status ?? 200;
    return { ...resp, response: { status, headers: resp.response?.headers ?? new Headers() } };
  };
  return {
    GET: async (p: string, i?: any) => withHeaders(routes.GET ? routes.GET(p, i) : fail("no GET")),
    POST: async (p: string, i?: any) => withHeaders(routes.POST ? routes.POST(p, i) : fail("no POST")),
    PUT: async (p: string, i?: any) => withHeaders(routes.PUT ? routes.PUT(p, i) : fail("no PUT")),
    DELETE: async (p: string, i?: any) => withHeaders(routes.DELETE ? routes.DELETE(p, i) : fail("no DELETE")),
  } as unknown as GalaxyClient;
}
