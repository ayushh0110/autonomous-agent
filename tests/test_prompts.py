"""End-to-end prompt tests — send real queries through the full pipeline.

Tests the complete flow: query → classifier → planner → agent → tools → response.
Requires the backend to be running (uses live LLM + tools).
"""

import sys
import os
import time
import json

# Fix Windows encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

API_URL = "http://localhost:8000/chat"
TIMEOUT = 60.0  # generous timeout for LLM calls

# ── Test cases ──────────────────────────────────────────────────────────
# Each test: (query, expected_keywords, expected_tool, description)
TEST_CASES = [
    # 1. Direct answer — greeting
    {
        "query": "hello how are you?",
        "expect_in_answer": ["hello", "doing"],
        "expect_tool": None,
        "description": "Greeting → direct LLM answer, no tools",
    },
    # 2. Calculator — basic math
    {
        "query": "what is 2+2?",
        "expect_in_answer": ["4"],
        "expect_tool": "calculator",
        "description": "Simple math → calculator tool",
    },
    # 3. Calculator — square root
    {
        "query": "what is the square root of 144?",
        "expect_in_answer": ["12"],
        "expect_tool": "calculator",
        "description": "Square root → calculator tool",
    },
    # 4. Translation
    {
        "query": "how do you say thank you in japanese?",
        "expect_in_answer": ["arigatou", "ありがとう", "arigatō", "arigato"],
        "expect_tool": "translate",
        "description": "Translation → translate tool",
    },
    # 5. Unit conversion
    {
        "query": "convert 100 fahrenheit to celsius",
        "expect_in_answer": ["37", "38"],
        "expect_tool": "unit_converter",
        "description": "Temperature conversion → unit_converter tool",
    },
    # 6. DateTime — timezone
    {
        "query": "what time is it in tokyo?",
        "expect_in_answer": ["tokyo", "time", ":"],
        "expect_tool": "datetime",
        "description": "Timezone query → datetime tool",
    },
    # 7. Dictionary
    {
        "query": "define serendipity",
        "expect_in_answer": ["finding", "unexpected", "fortunate", "chance", "discovery", "pleasant"],
        "expect_tool": "dictionary",
        "description": "Word definition → dictionary tool",
    },
    # 8. Wikipedia
    {
        "query": "tell me about Albert Einstein from wikipedia",
        "expect_in_answer": ["einstein", "physics", "relativity", "theoretical"],
        "expect_tool": "wikipedia",
        "description": "Knowledge lookup → wikipedia tool",
    },
    # 9. Knowledge — direct answer (no tool needed)
    {
        "query": "explain what photosynthesis is",
        "expect_in_answer": ["light", "plant", "energy", "sun", "carbon"],
        "expect_tool": None,
        "description": "Knowledge question → direct answer from LLM",
    },
    # 10. Web search
    {
        "query": "latest AI news today",
        "expect_in_answer": [],  # content varies
        "expect_tool": "web_search",
        "description": "Recency query → web_search tool",
    },
]


def send_query(query: str) -> dict:
    """Send a query to the chat endpoint and return the response."""
    payload = {"query": query}
    resp = httpx.post(API_URL, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def check_answer(answer: str, expected_keywords: list[str]) -> tuple[bool, list[str]]:
    """Check if the answer contains at least one expected keyword."""
    if not expected_keywords:
        return True, []
    answer_lower = answer.lower()
    found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    return len(found) > 0, found


def check_tool(tools_used: list[str], expected_tool: str | None) -> bool:
    """Check if the expected tool was used (or no tool expected)."""
    if expected_tool is None:
        return True  # No specific tool expected
    return expected_tool in tools_used


def run_tests():
    """Run all prompt tests and print a report."""
    print("=" * 70)
    print("  AUTONOMOUS AGENT — END-TO-END PROMPT TESTS")
    print("=" * 70)
    print()

    # Check if server is running
    try:
        httpx.get("http://localhost:8000/health", timeout=5.0)
    except Exception:
        print("❌ Backend not reachable at http://localhost:8000")
        print("   Make sure: uvicorn app.main:app --reload")
        sys.exit(1)

    results = []
    total_time = 0

    for i, test in enumerate(TEST_CASES, 1):
        query = test["query"]
        desc = test["description"]
        expect_kw = test["expect_in_answer"]
        expect_tool = test["expect_tool"]

        print(f"Test {i:2d}/10: {desc}")
        print(f"         Query: \"{query}\"")

        try:
            start = time.time()
            response = send_query(query)
            elapsed = time.time() - start
            total_time += elapsed

            answer = response.get("response", "")
            tools = response.get("tools_used", [])
            decision = response.get("decision", "")

            # Check answer content
            kw_pass, kw_found = check_answer(answer, expect_kw)

            # Check tool usage (secondary — warn only)
            tool_pass = check_tool(tools, expect_tool)

            # Overall pass: answer quality is primary, tool is secondary
            # If answer has expected content → pass (even if wrong tool)
            # If right tool was used → pass (tool gave correct data)
            passed = kw_pass or (tool_pass and expect_tool is not None)
            # If no keywords expected and no tool expected, pass if we got any answer
            if not expect_kw and expect_tool is None:
                passed = len(answer) > 0

            status = "PASS" if passed else "FAIL"
            icon = "+" if passed else "X"
            print(f"         [{icon}] {status}  ({elapsed:.1f}s)")

            if not kw_pass and expect_kw:
                print(f"         [!] Expected keywords: {expect_kw}")
                print(f"         [!] Answer preview: {answer[:120]}...")

            if not tool_pass and expect_tool:
                print(f"         [~] Tool mismatch: expected={expect_tool}, got={tools} (answer may still be correct)")

            print(f"         Decision: {decision} | Tools: {tools or 'none'}")
            print(f"         Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            print()

            results.append({
                "test": i,
                "description": desc,
                "passed": passed,
                "elapsed": elapsed,
                "tools_used": tools,
                "decision": decision,
                "answer_length": len(answer),
            })

        except Exception as exc:
            print(f"         ❌ ERROR: {exc}")
            print()
            results.append({
                "test": i,
                "description": desc,
                "passed": False,
                "elapsed": 0,
                "error": str(exc),
            })

        # Brief pause between tests to avoid rate limits
        time.sleep(2)

    # ── Summary ──
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])

    print("=" * 70)
    print(f"  RESULTS: {passed}/{len(results)} passed, {failed} failed")
    print(f"  Total time: {total_time:.1f}s (avg {total_time/len(results):.1f}s per test)")
    print("=" * 70)
    print()

    # Detailed summary table
    print(f"{'#':>3} {'Status':>8} {'Time':>6} {'Tools':>15} {'Description':<40}")
    print("-" * 75)
    for r in results:
        status = " [+]" if r["passed"] else " [X]"
        tools = ", ".join(r.get("tools_used", [])) or "none"
        elapsed = f"{r['elapsed']:.1f}s"
        print(f"{r['test']:3d} {status:>8} {elapsed:>6} {tools:>15} {r['description']:<40}")

    print()
    return passed == len(results)


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
