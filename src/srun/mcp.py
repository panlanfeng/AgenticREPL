"""MCP (Model Context Protocol) client — connects to external tool servers via JSON-RPC over stdio."""

import json
import subprocess
import threading
import time
import os


class MCPServer:
    """Manages a single MCP server connection over stdio."""

    def __init__(self, name, command, args=None, env=None):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self._process = None
        self._tools = []
        self._ready = threading.Event()
        self._response_lock = threading.Lock()
        self._pending = {}
        self._next_id = 0
        self._reader_thread = None
        self._connected = False

    def connect(self, timeout=10):
        """Start the MCP server process and perform handshake."""
        cmd_env = os.environ.copy()
        cmd_env.update(self.env)
        try:
            self._process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=cmd_env,
            )
        except FileNotFoundError:
            return False, f"MCP server '{self.name}': command '{self.command}' not found"
        except Exception as e:
            return False, f"MCP server '{self.name}': {e}"

        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

        # Initialize
        success, result = self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "AgenticREPL", "version": "0.1.0"},
        }, timeout=timeout)
        if not success:
            self._stop()
            return False, f"MCP init failed for '{self.name}': {result}"

        # Send initialized notification
        self._send_notification("notifications/initialized", {})

        # Discover tools
        success, tools_result = self._send_request("tools/list", {}, timeout=timeout)
        if not success:
            self._stop()
            return False, f"MCP tools/list failed for '{self.name}': {tools_result}"

        self._tools = tools_result.get("tools", [])
        self._connected = True
        return True, f"Connected to '{self.name}' ({len(self._tools)} tools)"

    def tools(self):
        """Return tool definitions in OpenAI-compatible format."""
        result = []
        for t in self._tools:
            params = t.get("inputSchema", {})
            result.append({
                "type": "function",
                "function": {
                    "name": f"mcp_{self.name}_{t['name']}",
                    "description": t.get("description", f"MCP tool: {self.name}/{t['name']}"),
                    "parameters": {
                        "type": "object",
                        "properties": params.get("properties", {}),
                    },
                },
            })
        return result

    def call_tool(self, tool_name, arguments):
        """Call an MCP tool by name (without mcp_ prefix)."""
        success, result = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        if not success:
            return f"MCP tool error ({self.name}/{tool_name}): {result}"
        content = result.get("content", [])
        if isinstance(content, list):
            return "\n".join(
                c.get("text", str(c)) for c in content if isinstance(c, dict)
            )
        return str(content)

    def disconnect(self):
        self._stop()

    def _send_request(self, method, params, timeout=30):
        request_id = self._next_id
        self._next_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        with self._response_lock:
            self._pending[request_id] = threading.Event()
        try:
            self._process.stdin.write(json.dumps(request) + "\n")
            self._process.stdin.flush()
        except Exception as e:
            return False, str(e)
        event = self._pending.get(request_id)
        if event:
            if event.wait(timeout):
                with self._response_lock:
                    result = self._pending.pop(request_id, None)
                if result and "error" in result:
                    return False, result["error"].get("message", str(result["error"]))
                return True, result.get("result", {}) if result else {}
        return False, "MCP timeout"

    def _send_notification(self, method, params):
        notification = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            self._process.stdin.write(json.dumps(notification) + "\n")
            self._process.stdin.flush()
        except Exception:
            pass

    def _read_loop(self):
        while self._process and self._process.poll() is None:
            try:
                line = self._process.stdout.readline()
                if not line:
                    break
                response = json.loads(line)
                req_id = response.get("id")
                if req_id is not None:
                    event = self._pending.get(req_id)
                    if event:
                        self._pending[req_id] = response
                        event.set()
            except (json.JSONDecodeError, Exception):
                time.sleep(0.05)

    def _stop(self):
        self._connected = False
        if self._process:
            try:
                self._process.stdin.close()
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None


class MCPManager:
    """Manages multiple MCP server connections."""

    def __init__(self):
        self._servers = {}

    def add_server(self, name, command, args=None, env=None):
        server = MCPServer(name, command, args, env)
        success, msg = server.connect()
        if not success:
            print(f"\033[2m  MCP {msg}\033[0m")
            return False
        self._servers[name] = server
        return True

    def all_tools(self):
        tools = []
        for server in self._servers.values():
            tools.extend(server.tools())
        return tools

    def call_tool(self, full_name, arguments):
        """full_name format: mcp_<server>_<tool_name>."""
        if not full_name.startswith("mcp_"):
            return f"Unknown MCP tool: {full_name}"
        parts = full_name[4:].split("_", 1)
        if len(parts) != 2:
            return f"Invalid MCP tool name: {full_name}"
        server_name, tool_name = parts
        server = self._servers.get(server_name)
        if not server:
            return f"MCP server '{server_name}' not connected"
        return server.call_tool(tool_name, arguments)

    def disconnect_all(self):
        for server in list(self._servers.values()):
            server.disconnect()
        self._servers.clear()


mcp = MCPManager()
