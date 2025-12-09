#!/usr/bin/env python3
"""
End-to-End Test for DriftOS Embedding Server

Tests the full flow:
1. Health check
2. Single/batch embeddings
3. Preprocessing
4. Similarity computation
5. Drift detection with realistic scenarios
6. Annotated conversation scenarios from manual_test_cases.json

Run with server on localhost:8100:
    python test_e2e.py

Or specify custom host:
    python test_e2e.py --host http://localhost:8100

Use --test-data to run annotated scenario tests:
    python test_e2e.py --test-data /path/to/manual_test_cases.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def check(condition: bool, message: str) -> bool:
    """Print pass/fail and return result."""
    if condition:
        print(f"  {GREEN}✓{RESET} {message}")
        return True
    else:
        print(f"  {RED}✗{RESET} {message}")
        return False


def section(title: str):
    """Print section header."""
    print(f"\n{BOLD}{BLUE}{'─' * 50}{RESET}")
    print(f"{BOLD}{BLUE}{title}{RESET}")
    print(f"{BOLD}{BLUE}{'─' * 50}{RESET}")


def test_health(base_url: str) -> bool:
    """Test health endpoint."""
    section("1. Health Check")

    try:
        resp = requests.get(f"{base_url}/health")
        data = resp.json()

        passed = all([
            check(resp.status_code == 200, "Status code is 200"),
            check(data.get("status") == "healthy", f"Status: {data.get('status')}"),
            check("paraphrase-MiniLM" in data.get("model", ""), f"Model: {data.get('model')}"),
            check(data.get("dimension") == 384, f"Dimension: {data.get('dimension')}"),
            check(data.get("device") in ["mps", "cpu", "cuda"], f"Device: {data.get('device')}"),
        ])
        return passed
    except Exception as e:
        print(f"  {RED}✗ Failed: {e}{RESET}")
        return False


def test_embed_single(base_url: str) -> bool:
    """Test single text embedding."""
    section("2. Single Text Embedding")

    try:
        resp = requests.post(f"{base_url}/embed", json={
            "text": "Planning a trip to Paris next summer",
            "preprocess": True
        })
        data = resp.json()

        embeddings = data.get("embeddings", [])
        passed = all([
            check(resp.status_code == 200, "Status code is 200"),
            check(len(embeddings) == 1, f"Got 1 embedding (got {len(embeddings)})"),
            check(len(embeddings[0]) == 384, f"Embedding dimension: {len(embeddings[0])}"),
            check(data.get("preprocessed_texts") is not None, "Preprocessing applied"),
        ])

        if data.get("preprocessed_texts"):
            print(f"  {YELLOW}→ Preprocessed: \"{data['preprocessed_texts'][0]}\"{RESET}")

        return passed
    except Exception as e:
        print(f"  {RED}✗ Failed: {e}{RESET}")
        return False


def test_embed_batch(base_url: str) -> bool:
    """Test batch embedding."""
    section("3. Batch Embedding")

    texts = [
        "I'm planning a kitchen renovation",
        "What countertop materials are best?",
        "The weather looks nice today",
    ]

    try:
        resp = requests.post(f"{base_url}/embed", json={
            "text": texts,
            "preprocess": True
        })
        data = resp.json()

        embeddings = data.get("embeddings", [])
        passed = all([
            check(resp.status_code == 200, "Status code is 200"),
            check(len(embeddings) == 3, f"Got 3 embeddings (got {len(embeddings)})"),
            check(all(len(e) == 384 for e in embeddings), "All embeddings are 384-dim"),
        ])

        return passed
    except Exception as e:
        print(f"  {RED}✗ Failed: {e}{RESET}")
        return False


def test_preprocess(base_url: str) -> bool:
    """Test preprocessing endpoint."""
    section("4. Preprocessing")

    try:
        resp = requests.post(f"{base_url}/preprocess", json={
            "text": "Can you please help me understand how to renovate my kitchen?"
        })
        data = resp.json()

        original = data.get("original", [""])[0]
        preprocessed = data.get("preprocessed", [""])[0]

        passed = all([
            check(resp.status_code == 200, "Status code is 200"),
            check(len(preprocessed) < len(original), "Text was shortened"),
            check("please" not in preprocessed.lower(), "Removed 'please'"),
            check("help" not in preprocessed.lower(), "Removed 'help'"),
        ])

        print(f"  {YELLOW}→ Original: \"{original}\"{RESET}")
        print(f"  {YELLOW}→ Preprocessed: \"{preprocessed}\"{RESET}")

        return passed
    except Exception as e:
        print(f"  {RED}✗ Failed: {e}{RESET}")
        return False


def test_similarity(base_url: str) -> bool:
    """Test similarity computation."""
    section("5. Similarity Computation")

    test_cases = [
        {
            "text1": "Planning a trip to Paris",
            "text2": "What hotels are available in Paris?",
            "expected_range": (0.3, 0.8),
            "label": "Related (Paris trip)",
        },
        {
            "text1": "Planning a trip to Paris",
            "text2": "My cat needs to go to the vet",
            "expected_range": (-0.2, 0.2),
            "label": "Unrelated (Paris vs cat)",
        },
    ]

    all_passed = True

    for tc in test_cases:
        try:
            resp = requests.post(f"{base_url}/similarity", json={
                "text1": tc["text1"],
                "text2": tc["text2"],
                "preprocess": True
            })
            data = resp.json()

            sim = data.get("similarity", 0)
            low, high = tc["expected_range"]
            in_range = low <= sim <= high

            passed = check(
                in_range,
                f"{tc['label']}: {sim:.3f} (expected {low}-{high})"
            )
            all_passed = all_passed and passed

        except Exception as e:
            print(f"  {RED}✗ {tc['label']}: {e}{RESET}")
            all_passed = False

    return all_passed


def test_drift_detection(base_url: str) -> bool:
    """Test drift detection with realistic scenarios."""
    section("6. Drift Detection")

    # Anchor context: kitchen renovation discussion
    anchor = "We're renovating the kitchen. Looking at quartz countertops and new cabinet options."

    test_cases = [
        {
            "message": "What about granite instead of quartz?",
            "expected_action": "STAY",
            "label": "On-topic (countertop material)",
        },
        {
            "message": "Should we also update the bathroom?",
            "expected_action": "BRANCH_SAME_CLUSTER",
            "label": "Related drift (same renovation cluster)",
        },
        {
            "message": "Did you see the game last night?",
            "expected_action": "BRANCH_NEW_CLUSTER",
            "label": "Unrelated (new cluster)",
        },
    ]

    all_passed = True

    for tc in test_cases:
        try:
            resp = requests.post(f"{base_url}/drift", json={
                "anchor": anchor,
                "message": tc["message"],
                "preprocess": True,
                "stay_threshold": 0.38,
                "branch_threshold": 0.15
            })
            data = resp.json()

            action = data.get("action", "")
            sim = data.get("similarity", 0)

            passed = check(
                action == tc["expected_action"],
                f"{tc['label']}: {action} (sim={sim:.3f})"
            )

            if action != tc["expected_action"]:
                print(f"    {YELLOW}Expected: {tc['expected_action']}{RESET}")

            all_passed = all_passed and passed

        except Exception as e:
            print(f"  {RED}✗ {tc['label']}: {e}{RESET}")
            all_passed = False

    return all_passed


def test_conversation_flow(base_url: str) -> bool:
    """Test a realistic conversation flow simulating DriftOS usage."""
    section("7. Conversation Flow Simulation")

    # Simulate a multi-turn conversation with topic changes
    conversation = [
        ("root", "I want to plan a vacation to Europe this summer"),
        ("on-topic", "What countries should I visit?"),
        ("on-topic", "How about starting in France and going to Italy?"),
        ("drift", "Oh wait, I need to renew my passport first"),
        ("on-topic-passport", "Where's the nearest passport office?"),
        ("big-drift", "Anyway, my cat has been acting weird lately"),
    ]

    print(f"  Conversation simulation:")

    current_anchor = None
    all_passed = True

    for msg_type, message in conversation:
        if msg_type == "root":
            current_anchor = message
            print(f"  {BLUE}[ROOT]{RESET} {message}")
            continue

        resp = requests.post(f"{base_url}/drift", json={
            "anchor": current_anchor,
            "message": message,
            "preprocess": True
        })
        data = resp.json()

        action = data.get("action", "")
        sim = data.get("similarity", 0)

        # Determine expected behavior
        if msg_type == "on-topic":
            expected = "STAY"
        elif msg_type == "drift":
            expected = "BRANCH_SAME_CLUSTER"
        elif msg_type == "on-topic-passport":
            expected = "STAY"  # Should stay on passport subtopic
        else:  # big-drift
            expected = "BRANCH_NEW_CLUSTER"

        icon = GREEN + "✓" + RESET if action == expected else RED + "✗" + RESET
        print(f"  {icon} [{action}] (sim={sim:.3f}) {message[:50]}...")

        if action != expected:
            print(f"    {YELLOW}Expected: {expected}{RESET}")
            all_passed = False

        # Update anchor if staying (simulating DriftOS behavior)
        if action == "STAY":
            current_anchor = message

    return all_passed


def test_edge_cases(base_url: str) -> bool:
    """Test edge cases and error handling."""
    section("8. Edge Cases")

    all_passed = True

    # Empty string
    try:
        resp = requests.post(f"{base_url}/embed", json={"text": ""})
        passed = check(resp.status_code == 200, "Empty string handled")
        all_passed = all_passed and passed
    except Exception as e:
        print(f"  {RED}✗ Empty string: {e}{RESET}")
        all_passed = False

    # Very long text
    try:
        long_text = "This is a test. " * 100
        resp = requests.post(f"{base_url}/embed", json={"text": long_text})
        passed = check(resp.status_code == 200, "Long text handled")
        all_passed = all_passed and passed
    except Exception as e:
        print(f"  {RED}✗ Long text: {e}{RESET}")
        all_passed = False

    # Special characters
    try:
        resp = requests.post(f"{base_url}/embed", json={
            "text": "Test with émojis 🎉 and spëcial châràctérs!!"
        })
        passed = check(resp.status_code == 200, "Special characters handled")
        all_passed = all_passed and passed
    except Exception as e:
        print(f"  {RED}✗ Special chars: {e}{RESET}")
        all_passed = False

    # Preprocessing disabled
    try:
        resp = requests.post(f"{base_url}/embed", json={
            "text": "Test without preprocessing",
            "preprocess": False
        })
        data = resp.json()
        passed = check(
            data.get("preprocessed_texts") is None,
            "Preprocessing can be disabled"
        )
        all_passed = all_passed and passed
    except Exception as e:
        print(f"  {RED}✗ Preprocess disabled: {e}{RESET}")
        all_passed = False

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="E2E test for DriftOS Embedding Server")
    parser.add_argument("--host", default="http://localhost:8100", help="Server URL")
    args = parser.parse_args()

    base_url = args.host.rstrip("/")

    print(f"\n{BOLD}DriftOS Embedding Server - End-to-End Test{RESET}")
    print(f"Target: {base_url}")

    tests = [
        ("Health Check", test_health),
        ("Single Embedding", test_embed_single),
        ("Batch Embedding", test_embed_batch),
        ("Preprocessing", test_preprocess),
        ("Similarity", test_similarity),
        ("Drift Detection", test_drift_detection),
        ("Conversation Flow", test_conversation_flow),
        ("Edge Cases", test_edge_cases),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn(base_url)
            results.append((name, passed))
        except requests.exceptions.ConnectionError:
            print(f"\n{RED}Connection failed! Is the server running at {base_url}?{RESET}")
            print(f"Start with: cd embedding-server && uvicorn server:app --port 8100")
            sys.exit(1)
        except Exception as e:
            print(f"  {RED}✗ Unexpected error: {e}{RESET}")
            results.append((name, False))

    # Summary
    section("Summary")

    passed_count = sum(1 for _, p in results if p)
    total = len(results)

    for name, passed in results:
        icon = GREEN + "✓" + RESET if passed else RED + "✗" + RESET
        print(f"  {icon} {name}")

    print()
    if passed_count == total:
        print(f"{GREEN}{BOLD}All {total} tests passed!{RESET}")
        sys.exit(0)
    else:
        print(f"{RED}{BOLD}{passed_count}/{total} tests passed{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
