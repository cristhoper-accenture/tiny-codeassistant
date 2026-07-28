#!/usr/bin/env bash
# preload_models.sh — warm up all agent models into Ollama memory.
#
# Sends a keep_alive=-1 generate request for each unique model so they stay
# resident in RAM for the session. Run this once before starting the agent
# to eliminate cold-start latency on the first inference call.
#
# Usage:
#   ./preload_models.sh                  # load Preset A defaults
#   ./preload_models.sh --unload         # release all models from memory
#   CODER_MODEL=qwen2.5-coder:3b ./preload_models.sh  # respects env overrides

set -euo pipefail

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

# ── Resolve models (mirrors config.py Preset A defaults) ─────────────────────
ORCHESTRATOR_MODEL="${ORCHESTRATOR_MODEL:-qwen2.5:0.5b}"
CODER_MODEL="${CODER_MODEL:-qwen2.5-coder:7b}"
LINT_MODEL="${LINT_MODEL:-qwen2.5-coder:7b}"
PLANNER_MODEL="${PLANNER_MODEL:-phi4-mini:3.8b}"
GENERAL_MODEL="${GENERAL_MODEL:-phi4-mini:3.8b}"

# Collect unique model names
declare -A _seen
MODELS=()
for m in "$ORCHESTRATOR_MODEL" "$CODER_MODEL" "$LINT_MODEL" "$PLANNER_MODEL" "$GENERAL_MODEL"; do
    if [[ -z "${_seen[$m]+x}" ]]; then
        _seen[$m]=1
        MODELS+=("$m")
    fi
done

# ── Unload mode ───────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--unload" ]]; then
    echo "Unloading models from Ollama memory..."
    for model in "${MODELS[@]}"; do
        printf "  Unloading %-30s" "$model..."
        resp=$(curl -s -X POST "$OLLAMA_BASE_URL/api/generate" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"$model\", \"keep_alive\": 0}" 2>&1) || true
        echo "done"
    done
    echo "All models unloaded."
    exit 0
fi

# ── Check Ollama is reachable ─────────────────────────────────────────────────
if ! curl -sf "$OLLAMA_BASE_URL/api/tags" > /dev/null 2>&1; then
    echo "ERROR: Ollama is not reachable at $OLLAMA_BASE_URL"
    echo "Start it with: ollama serve"
    exit 1
fi

# ── Pull any missing models ───────────────────────────────────────────────────
echo "Checking models..."
available=$(curl -s "$OLLAMA_BASE_URL/api/tags" | python3 -c \
    "import sys,json; [print(m['name']) for m in json.load(sys.stdin).get('models',[])]" 2>/dev/null)

for model in "${MODELS[@]}"; do
    if ! echo "$available" | grep -qF "$model"; then
        echo "  Pulling $model (not found locally)..."
        ollama pull "$model"
    else
        echo "  $model — already available"
    fi
done

# ── Preload models in parallel ────────────────────────────────────────────────
echo ""
echo "Preloading ${#MODELS[@]} unique model(s) into memory (keep_alive=-1)..."
echo ""

_pids=()
for model in "${MODELS[@]}"; do
    (
        printf "  Loading %-30s" "$model..."
        start=$SECONDS
        resp=$(curl -s -X POST "$OLLAMA_BASE_URL/api/generate" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"$model\", \"keep_alive\": -1}" \
            --max-time 300 2>&1)
        elapsed=$(( SECONDS - start ))
        if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if not d.get('error') else 1)" 2>/dev/null; then
            echo "ready  (${elapsed}s)"
        else
            echo "ERROR: $resp"
        fi
    ) &
    _pids+=($!)
done

# Wait for all parallel loads
for pid in "${_pids[@]}"; do
    wait "$pid"
done

echo ""
echo "All models loaded. RAM usage:"
curl -s "$OLLAMA_BASE_URL/api/ps" | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = data.get('models', [])
if not models:
    print('  (no models reported by /api/ps)')
else:
    total = 0
    for m in models:
        size = m.get('size', 0)
        total += size
        vram = m.get('size_vram', 0)
        ram  = size - vram
        print(f'  {m[\"name\"]:<35} RAM: {ram/1024**3:.1f} GB  VRAM: {vram/1024**3:.1f} GB')
    print(f'  {\"TOTAL\":<35} {total/1024**3:.1f} GB')
" 2>/dev/null || echo "  (install python3 to see RAM breakdown)"

echo ""
echo "Ready. Run the agent with: ./run.sh"
