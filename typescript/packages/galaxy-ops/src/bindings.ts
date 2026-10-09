// THE single bindings source: Galaxy's published OpenAPI bindings on npm.
// galaxy-ops consumes the TYPES only (components, GalaxyApiPaths) and builds its own
// openapi-fetch client (see client.ts). Bump @galaxyproject/galaxy-api-client per the
// targeted Galaxy release (currently 26.0.x; move to 26.1.x once it publishes).
export type { components, GalaxyApiPaths } from "@galaxyproject/galaxy-api-client";

import type { components } from "@galaxyproject/galaxy-api-client";
export type Schemas = components["schemas"];

import type { GalaxyApiPaths } from "@galaxyproject/galaxy-api-client";

/** The 200 application/json response body of a GET path. */
export type GetJson<P extends keyof GalaxyApiPaths> =
  GalaxyApiPaths[P] extends { get: { responses: { 200: { content: { "application/json": infer R } } } } }
    ? R : never;

/** The 200/201 application/json response body of a POST path. */
export type PostJson<P extends keyof GalaxyApiPaths> =
  GalaxyApiPaths[P] extends { post: { responses: infer Rs } }
    ? Rs extends { 200: { content: { "application/json": infer R } } } ? R
      : Rs extends { 201: { content: { "application/json": infer R } } } ? R
      : never
    : never;

/** The 200 application/json response body of a PUT path. */
export type PutJson<P extends keyof GalaxyApiPaths> =
  GalaxyApiPaths[P] extends { put: { responses: { 200: { content: { "application/json": infer R } } } } }
    ? R : never;
