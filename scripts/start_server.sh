#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# start flask server
cd "$SCRIPT_DIR/../flask-server"
echo "Starting Flask server..."
python server.py &  # Run Flask in the background

# start node server
cd "$SCRIPT_DIR/../client"
echo "Starting Node.js server..."
npm start &  # Run Node.js server in the background

# Wait to keep the script running
wait
