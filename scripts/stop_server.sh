#!/bin/bash

echo "Stopping Flask and Node.js servers..."

# Kill Flask server 
pkill -f "python server.py"

# Kill Node server
pkill -f "node"

echo "Servers stopped."
