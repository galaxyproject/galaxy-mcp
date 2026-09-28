"""Reusable Galaxy operation logic, independent of MCP transport and Galaxy I/O.

Pure functions of their arguments: no client, no network, no global state, nothing that
knows an MCP server exists. Fetching, session handling and tool registration belong to the
caller, which is what lets this be read and tested on its own, and what the TypeScript ports
in galaxy-agent-tools follow.

Input contracts are what lives here today. Anything else that turns out to be reusable
Galaxy logic rather than transport or orchestration belongs here too.

Nothing is imported here, so a caller pays only for the module it names.
"""
