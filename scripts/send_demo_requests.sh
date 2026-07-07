#!/usr/bin/env bash
set -euo pipefail

URL="${1:-http://localhost:8000/predict}"
N="${N:-40}"

echo "Sending $N demo prediction requests to $URL"

for i in $(seq 1 "$N"); do
  case $((i % 4)) in
    0)
      PAYLOAD='{
        "text": "Mbappe scored two goals in the World Cup final. Unreal player.",
        "candidate_player": "Kylian Mbappé",
        "position": "Forward"
      }'
      ;;
    1)
      PAYLOAD='{
        "text": "Ronaldo was terrible this World Cup and offered nothing for Portugal.",
        "candidate_player": "Cristiano Ronaldo",
        "position": "Forward"
      }'
      ;;
    2)
      PAYLOAD='{
        "text": "Benzema will miss the World Cup due to injury.",
        "candidate_player": "Karim Benzema",
        "position": "Forward"
      }'
      ;;
    3)
      PAYLOAD='{
        "text": "Bellingham is carrying England again. Unreal player.",
        "candidate_player": "Jude Bellingham",
        "position": "Midfielder"
      }'
      ;;
  esac

  curl -s -X POST "$URL" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" > /dev/null

  echo "sent request $i/$N"
  sleep 0.2
done

echo "Done."