#!/bin/bash
# Test script for LLM Metrics Proxy
# This demonstrates how the proxy works with snippet-grounded mode

echo "=========================================="
echo "LLM Metrics Proxy Test Script"
echo "=========================================="

# Check if proxy is running
echo ""
echo "1. Checking proxy health..."
curl -s http://localhost:8000/health | python3 -m json.tool

# List problems
echo ""
echo "2. Listing configured problems..."
curl -s http://localhost:8000/api/problems | python3 -m json.tool

# Test a request (you need to set OPENAI_API_KEY)
echo ""
echo "3. Testing LLM request through proxy..."
echo "   (Requires OPENAI_API_KEY environment variable)"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "   ⚠️  OPENAI_API_KEY not set. Skipping LLM test."
    echo "   Set it with: export OPENAI_API_KEY='your-key'"
else
    echo "   Sending request to binary_search problem..."
    
    curl -s -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $OPENAI_API_KEY" \
      -H "X-Problem-ID: binary_search" \
      -H "X-Candidate-ID: test_user" \
      -d '{
        "model": "gpt-4o-mini",
        "messages": [
          {"role": "user", "content": "How do I implement the binary_search function?"}
        ],
        "max_tokens": 500
      }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'choices' in data:
    print('Response:', data['choices'][0]['message']['content'][:500], '...')
    print()
    print('Usage:', data.get('usage', {}))
else:
    print('Error:', data)
"
fi

# Check metrics
echo ""
echo "4. Checking captured metrics..."
curl -s http://localhost:8000/api/metrics/overview | python3 -m json.tool

# Mode comparison
echo ""
echo "5. Mode comparison (snippet_grounded vs free_form)..."
curl -s http://localhost:8000/api/metrics/mode-comparison | python3 -m json.tool

echo ""
echo "=========================================="
echo "Test complete!"
echo ""
echo "To configure Cline to use this proxy:"
echo "  1. Open VS Code Settings (Cmd+,)"
echo "  2. Search for 'Cline'"
echo "  3. Set these values:"
echo "     - API Provider: openai"
echo "     - OpenAI Base URL: http://localhost:8000/v1"
echo "     - API Key: your-openai-key"
echo ""
echo "Toggle mode with:"
echo "  curl -X POST http://localhost:8000/api/problems/binary_search/toggle"
echo "=========================================="
