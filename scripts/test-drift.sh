#!/bin/bash
# DriftOS Full API Test Script
# Tests messages, conversations, branches, and context endpoints

set -e

API_URL="${API_URL:-http://localhost:3001/api/v1}"
CONV_ID="test-$(date +%s)"
BRANCH_ID=""

echo "🧪 DriftOS Full API Test"
echo "========================"
echo "API: $API_URL"
echo "Conversation: $CONV_ID"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Helper: send message
send_message() {
  local content="$1"
  local use_branch="$2"
  local expected="$3"
  
  echo -e "${BLUE}📤 Message:${NC} \"$content\""
  
  if [ -z "$use_branch" ]; then
    payload="{\"conversationId\": \"$CONV_ID\", \"content\": \"$content\"}"
  else
    payload="{\"conversationId\": \"$CONV_ID\", \"content\": \"$content\", \"currentBranchId\": \"$use_branch\"}"
  fi
  
  response=$(curl -s -X POST "$API_URL/messages" \
    -H "Content-Type: application/json" \
    -d "$payload")
  
  success=$(echo "$response" | jq -r '.success')
  if [ "$success" != "true" ]; then
    error=$(echo "$response" | jq -r '.error.message // .error // "Unknown error"')
    echo -e "   ${RED}❌ Error: $error${NC}"
    echo ""
    return 1
  fi
  
  action=$(echo "$response" | jq -r '.data.action')
  drift_action=$(echo "$response" | jq -r '.data.driftAction')
  new_branch_id=$(echo "$response" | jq -r '.data.branchId')
  similarity=$(echo "$response" | jq -r '.data.similarity')
  is_new_branch=$(echo "$response" | jq -r '.data.isNewBranch')
  
  BRANCH_ID="$new_branch_id"
  
  case "$drift_action" in
    "STAY") color=$GREEN ;;
    "BRANCH_SAME_CLUSTER") color=$YELLOW ;;
    "BRANCH_NEW_CLUSTER") color=$RED ;;
    *) color=$NC ;;
  esac
  
  echo -e "   ${color}▶ $action ($drift_action)${NC} | Similarity: $similarity | New: $is_new_branch"
  
  if [ -n "$expected" ] && [ "$drift_action" != "$expected" ]; then
    echo -e "   ${RED}⚠️  Expected: $expected${NC}"
  fi
  echo ""
}

# ═══════════════════════════════════════════════════════════════════
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}  PART 1: DRIFT DETECTION                                      ${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
echo ""

echo "━━━ Test 1: First message (creates branch) ━━━"
send_message "I want to book a hotel in Paris for my trip next month" "" "BRANCH_NEW_CLUSTER"

echo "━━━ Test 2: Paraphrase (should STAY) ━━━"
send_message "Looking to reserve accommodation in Paris for next month" "$BRANCH_ID" "STAY"

echo "━━━ Test 3: Same subtopic (should STAY) ━━━"
send_message "What's the average price for hotels near the Eiffel Tower?" "$BRANCH_ID" "STAY"

PARIS_BRANCH="$BRANCH_ID"

echo "━━━ Test 4: Different domain (should BRANCH_NEW_CLUSTER) ━━━"
send_message "How do I fix a Python memory leak?" "$BRANCH_ID" "BRANCH_NEW_CLUSTER"

echo "━━━ Test 5: Stay in Python topic ━━━"
send_message "What's the best way to profile memory in Python?" "$BRANCH_ID" "STAY"

PYTHON_BRANCH="$BRANCH_ID"

# ═══════════════════════════════════════════════════════════════════
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}  PART 2: CONVERSATION ENDPOINTS                               ${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${CYAN}GET /conversations/$CONV_ID${NC}"
curl -s "$API_URL/conversations/$CONV_ID" | jq '.data | {branchCount, messageCount, currentBranchId}'
echo ""

echo -e "${CYAN}GET /conversations/$CONV_ID/branches${NC}"
curl -s "$API_URL/conversations/$CONV_ID/branches" | jq '.data | map({id: .id[0:12], topic: .topic[0:40], messages: .messageCount, depth})'
echo ""

echo -e "${CYAN}GET /conversations/$CONV_ID/context${NC}"
curl -s "$API_URL/conversations/$CONV_ID/context" | jq '{
  branches: (.data.branches | map(.topic[0:30])),
  messageCount: .data.stats.totalMessages,
  factCount: .data.stats.totalFacts
}'
echo ""

# ═══════════════════════════════════════════════════════════════════
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}  PART 3: BRANCH ENDPOINTS                                     ${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${CYAN}GET /branches/$PARIS_BRANCH${NC}"
curl -s "$API_URL/branches/$PARIS_BRANCH" | jq '.data | {topic, messageCount, factCount, depth}'
echo ""

echo -e "${CYAN}GET /branches/$PARIS_BRANCH/context${NC}"
curl -s "$API_URL/branches/$PARIS_BRANCH/context" | jq '{
  topic: .data.branchTopic,
  messages: (.data.messages | length),
  facts: (.data.facts | length)
}'
echo ""

echo -e "${CYAN}POST /branches/$PARIS_BRANCH/facts (extract)${NC}"
result=$(curl -s -X POST "$API_URL/branches/$PARIS_BRANCH/facts")
echo "$result" | jq '{extractedCount: .data.extractedCount, facts: [.data.facts[]?.key]}'
echo ""

echo -e "${CYAN}GET /branches/$PARIS_BRANCH/facts${NC}"
curl -s "$API_URL/branches/$PARIS_BRANCH/facts" | jq '.data | map({key, value, confidence, sources: (.messageIds | length)})'
echo ""

# ═══════════════════════════════════════════════════════════════════
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}  SUMMARY                                                      ${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Conversation: $CONV_ID"
echo "Paris Branch: ${PARIS_BRANCH:0:20}..."
echo "Python Branch: ${PYTHON_BRANCH:0:20}..."
echo ""
echo "Endpoints tested:"
echo "  POST /messages                    ✓"
echo "  GET  /conversations/:id           ✓"
echo "  GET  /conversations/:id/branches  ✓"
echo "  GET  /conversations/:id/context   ✓"
echo "  GET  /branches/:id                ✓"
echo "  GET  /branches/:id/context        ✓"
echo "  POST /branches/:id/facts          ✓"
echo "  GET  /branches/:id/facts          ✓"
echo ""
echo -e "${GREEN}✅ All tests complete!${NC}"
