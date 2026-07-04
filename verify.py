import subprocess
import time
import os
import re
import atexit
import socket
import portpicker
import nest_asyncio
from tqdm import tqdm
from client import Lean4Client


# ── Lean output cleaning (before verification) ───────────────────────────────

_LEADING_LANG_TAGS = {
    "tactics", "lean", "lean4", "leanprover", "mathlib", "math",
}


def sanitize_lean(code: str) -> str:
    """Peel leading markdown language-tag lines from model output (idempotent).

    Outputs that begin with a fenced ``\\`\\`\\`tactics`` block can leave the
    literal token ``tactics`` as the first line, which Lean rejects as
    ``unexpected identifier``. Strip any such leading tag line(s). No-op for
    clean inputs; line structure is otherwise preserved (only the BOM is
    removed) so compiling proofs keep their line/column offsets.
    """
    if not code:
        return code
    s = code.lstrip("﻿")  # strip BOM only — preserve line structure

    while True:
        rest = s.lstrip("\n\r\t ")
        if rest is s and not s:
            break
        nl = rest.find("\n")
        first = rest[:nl] if nl != -1 else rest
        if first.strip().lower() in _LEADING_LANG_TAGS:
            s = rest[nl + 1:] if nl != -1 else ""
        else:
            break
    return s


def clean_up_lean(code: str) -> str:
    """Strip markdown fences / think-header tags and normalize spacing."""
    code = code.replace("```lean", "")
    code = code.replace("```", "")
    code = code.replace("<think>", "")
    code = code.replace("</think>", "")
    code = code.replace("<header>", "")
    code = code.replace("</header>", "")
    code = code.replace("set option", "set_option")
    while "  := by" in code:
        code = code.replace("  := by", " := by")
    return code.strip()


# ── Lean-region extraction (for FR/RR/OUR judging) ───────────────────────────

_COMMENT_RE = re.compile(r"/-[\s\S]*?-/")
_BY_RE = re.compile(r":=\s*by\b")
_DECL_RE = re.compile(r"(?:theorem|lemma|example)\b")


def split_signature_proof(lean: str) -> tuple[str, str]:
    """Return ``(signature, proof_body)`` for a Lean 4 declaration.

    Strips the leading ``/- ... -/`` NL-comment block, locates the first
    ``theorem|lemma|example`` declaration, and splits at ``:= by``. Either
    component is "" if missing.
    """
    if not lean:
        return "", ""
    stripped = _COMMENT_RE.sub("", lean)
    m = _DECL_RE.search(stripped)
    if not m:
        return "", ""
    region = stripped[m.start():]
    bm = _BY_RE.search(region)
    if bm:
        return region[:bm.start()], region[bm.end():]
    return region, ""


def extract_region(lean_output: str, edit_source: str) -> str:
    """Return the slice the FR/RR/OUR judge should see, or "" if degenerate.

    ``edit_source`` is ``"statement"`` or ``"proof"``. Callers should treat an
    empty result as a deterministic "other" (no judge call needed).
    """
    sig, proof = split_signature_proof(lean_output)
    target = sig if edit_source == "statement" else proof
    return target.strip()


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Invalid {name}={value!r}, using default {default}")
        return default


def verify_lean(codes, kimina_batch_size=20480, timeout=300, lean_port=None, max_retries=5):
    nest_asyncio.apply()
    kimina_batch_size = _env_int("KIMINA_BATCH_SIZE", kimina_batch_size)
    timeout = _env_int("KIMINA_TIMEOUT", timeout)
    max_retries = _env_int("KIMINA_MAX_RETRIES", max_retries)
    print(
        f"verify_lean settings: batch_size={kimina_batch_size}, "
        f"timeout={timeout}s, max_retries={max_retries} "
    )

    for attempt in range(max_retries):
        server_process = None
        if lean_port is None:
            server_process, current_port = start_kimina_server()
        else:
            current_port = lean_port

        try:
            client = Lean4Client(base_url=f"http://127.0.0.1:{current_port}")

            final_results = []
            num_batches = (len(codes) + kimina_batch_size - 1) // kimina_batch_size
            for i in tqdm(range(num_batches)):
                batch = codes[i*kimina_batch_size:(i+1)*kimina_batch_size]
                requests = [
                    {"proof": clean_up_lean(sanitize_lean(code)), "custom_id": str(i)}
                    for i, code in enumerate(batch)
                ]
                responses = client.verify(requests, timeout=timeout)
                assert len(responses["results"]) == len(requests)
                results = {r["custom_id"]: r for r in responses["results"]}

                for request in requests:
                    result = results[request["custom_id"]]
                    response = result["response"]
                    ret = {
                        "system_error": result["error"],
                        "_response": response,
                        "code": request["proof"],
                    }
                    if not response:
                        print("no response!", str(result)[:1000])
                        assert result["error"]
                        ret.update({
                            "errors": [],
                            "sorries": [],
                            "time": None,
                        })
                    else:
                        ret.update({
                            "errors": [message for message in response.get("messages", [])
                                        if message and message["severity"] == "error"],
                            "sorries": response.get("sorries", []),
                            "time": response["time"],
                        })
                    ret["passed"] = not ret["errors"] and not ret["system_error"]
                    ret["complete"] = ret["passed"] and not ret["sorries"]
                    final_results.append(ret)

            return final_results
        except Exception as e:
            print(f"Server connection/verification failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
        finally:
            if server_process is not None:
                stop_kimina_server(server_process)


def start_kimina_server():
    port = portpicker.pick_unused_port()
    server_env = {**os.environ, "LEANSERVER_PORT": str(port)}
    # The server defaults to pre-filling one REPL per CPU. On large Slurm
    # nodes this can mean dozens of Mathlib REPLs before the health endpoint is
    # reachable, which makes short startup probes falsely fail or hang. Create
    # REPLs lazily during /verify instead.
    server_env.setdefault("LEANSERVER_PREFILL_REPLS", "0")
    server_env.setdefault("LEANSERVER_MAX_REPLS", "1")
    server_env.setdefault("LEANSERVER_MAX_CONCURRENT_REQUESTS", "1")
    server_env.setdefault("LEANSERVER_MAX_WORKERS", "4")
    process = subprocess.Popen(
        ["python", "-m", "server"],
        cwd="kimina-lean-server", env=server_env,
    )
    sleep_s = int(os.environ.get("LEANSERVER_STARTUP_SLEEP", "3"))
    requested_max_wait_s = int(os.environ.get("LEANSERVER_STARTUP_MAX_WAIT", "180"))
    max_wait_s = max(0, requested_max_wait_s)

    initial_sleep_s = min(max(0, sleep_s), max_wait_s)
    if initial_sleep_s:
        time.sleep(initial_sleep_s)

    connected = False
    deadline = time.time() + max(0.0, max_wait_s - initial_sleep_s)
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                connected = True
                break
        except OSError:
            time.sleep(0.5)
    if not connected:
        raise RuntimeError(f"Lean server did not become reachable within {max_wait_s}s (port={port}).")

    atexit.register(process.terminate)
    return process, port


def stop_kimina_server(process):
    process.terminate()
    process.wait()