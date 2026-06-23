#!/bin/bash
# Download React, ReactDOM, and Babel locally (avoids CDN dependency)
set -e
mkdir -p frontend/lib
curl -sL -o frontend/lib/react.min.js     https://unpkg.com/react@18/umd/react.production.min.js
curl -sL -o frontend/lib/react-dom.min.js https://unpkg.com/react-dom@18/umd/react-dom.production.min.js
curl -sL -o frontend/lib/babel.min.js     https://unpkg.com/@babel/standalone/babel.min.js
echo "Frontend libs ready."
