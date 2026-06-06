// THE single swap point for the bindings source.
//   today:    vendored generated schema (./generated)
//   tomorrow: export type { components, GalaxyApiPaths } from "@galaxyproject/galaxy-api-client";
// Nothing else in the package imports ./generated directly.
export type { components, GalaxyApiPaths } from "./generated/index";

import type { components } from "./generated/index";
export type Schemas = components["schemas"];
