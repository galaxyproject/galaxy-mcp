// Variables the code under test never reads itself but Node does: the proxy and TLS settings. With
// NODE_USE_ENV_PROXY=1 and a proxy set, fetch goes to the proxy even for a test server on
// 127.0.0.1, and a CA or cert override changes what TLS trusts, so a test's result would depend on
// the machine it runs on. Shared by global-setup.ts, which clears them before the workers start
// (Node reads NODE_USE_ENV_PROXY once, at startup, so clearing it later is too late), and
// safe-env.ts, which clears them again in each worker.
export const LEAKY_ENV = [
  "HTTP_PROXY",
  "HTTPS_PROXY",
  "ALL_PROXY",
  "NO_PROXY",
  "http_proxy",
  "https_proxy",
  "all_proxy",
  "no_proxy",
  "NODE_USE_ENV_PROXY",
  "NODE_EXTRA_CA_CERTS",
  "SSL_CERT_FILE",
  "SSL_CERT_DIR",
] as const;
