#!/usr/bin/env bash
# Quick smoke test against a running server (default http://127.0.0.1:8000)
set -euo pipefail
BASE="${1:-http://127.0.0.1:8000}"

echo "== Health =="
curl -sf "$BASE/health" | python3 -m json.tool

questions=(
  'user_101|Should I consider changing my job in the next few months?'
  'user_101|How does this month look for my relationship?'
  'user_102|What should I focus on for my health?'
  'user_103|What should I prioritize this week?'
  'user_101|Can you summarize today'\''s guidance?'
)

for entry in "${questions[@]}"; do
  uid="${entry%%|*}"
  q="${entry#*|}"
  echo
  echo "== personalize: $q =="
  curl -sf -X POST "$BASE/personalize" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c "import json; print(json.dumps({'userId':'$uid','question':'''$q'''}))")" \
    | python3 -m json.tool
done

echo
echo "== debug personalization (career) =="
curl -sf -X POST "$BASE/debug/personalization" \
  -H 'Content-Type: application/json' \
  -d '{"userId":"user_101","question":"Should I change my job?"}' \
  | python3 -m json.tool

echo
echo "All smoke tests passed."
