import type { GalaxyClient } from "../../src/client";

type Resp = { data?: unknown; error?: unknown; response?: { status: number } };
type Handler = (path: string, init?: any) => Resp | Promise<Resp>;

/** Hand-rolled GalaxyClient stand-in. We only ever call .GET/.POST. */
export function mockClient(routes: { GET?: Handler; POST?: Handler }): GalaxyClient {
  const fail = (m: string): Resp => ({ error: m, response: { status: 500 } });
  return {
    GET: async (p: string, i?: any) => (routes.GET ? routes.GET(p, i) : fail("no GET")),
    POST: async (p: string, i?: any) => (routes.POST ? routes.POST(p, i) : fail("no POST")),
  } as unknown as GalaxyClient;
}
