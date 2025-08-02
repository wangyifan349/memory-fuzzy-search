#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from flask import Flask, request, jsonify, send_from_directory, render_template_string
# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
# Create the Flask application object
application = Flask(__name__)
# Absolute path to this script’s directory
PROJECT_DIRECTORY = os.path.abspath(os.path.dirname(__file__))
# Directory where files are stored (must exist or will be created at startup)
FILES_DIRECTORY = os.path.join(PROJECT_DIRECTORY, 'files')
# ──────────────────────────────────────────────────────────────────────────────
# SINGLE HTML TEMPLATE STRING
# ──────────────────────────────────────────────────────────────────────────────
# The entire front-end HTML (Bootstrap 5 + AJAX) lives here in one variable
HTML_TEMPLATE = r'''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>File Search and Download</title>
  <!-- Bootstrap 5 CSS from CDN -->
  <link
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
    rel="stylesheet"
    integrity="sha384-ENjdO4Dr2bkBIFxQpeo9qU0F3sRJoQzUv4bYlVx8ELvYjZE0Nfpu6zWLAjrYcYUn"
    crossorigin="anonymous"
  />
</head>
<body>
  <div class="container py-5">
    <h1 class="text-center mb-4">File Search and Download</h1>

    <!-- Search Form -->
    <form id="fileSearchForm" class="row g-2 justify-content-center mb-4">
      <div class="col-12 col-md-8">
        <input
          type="text"
          id="searchQueryInput"
          class="form-control"
          placeholder="Enter keywords to search files..."
          aria-label="Search Query"
          required
        />
      </div>
      <div class="col-12 col-md-2">
        <button type="submit" class="btn btn-primary w-100">
          Search
        </button>
      </div>
    </form>

    <!-- Search Results Container -->
    <div id="searchResultsContainer" class="list-group"></div>
  </div>

  <!-- Bootstrap 5 JS Bundle (includes Popper) -->
  <script
    src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"
    integrity="sha384-q2gyF0pf6IQjHLF+8qLm4wBoyqoGnQZ9dq0U3cGFTF1uL7ox84Ca0F8qjq2OeRiP"
    crossorigin="anonymous"
  ></script>

  <script>
    // Wait until DOM content is loaded
    document.addEventListener('DOMContentLoaded', function () {
      const searchForm       = document.getElementById('fileSearchForm');
      const searchInput      = document.getElementById('searchQueryInput');
      const resultsContainer = document.getElementById('searchResultsContainer');

      // Handle form submission
      searchForm.addEventListener('submit', function (event) {
        event.preventDefault();  // Prevent full-page reload

        const searchQuery = searchInput.value.trim();
        if (!searchQuery) {
          alert('Please enter a search keyword.');
          return;
        }

        // Clear previous results
        resultsContainer.innerHTML = '';

        // Call the back-end API
        fetch(`/api/search_files?query=${encodeURIComponent(searchQuery)}`)
          .then(function (response) {
            if (!response.ok) {
              throw new Error(`Network response not ok (${response.status})`);
            }
            return response.json();
          })
          .then(function (jsonData) {
            if (jsonData.status !== 0) {
              throw new Error('API returned error status');
            }
            renderSearchResults(jsonData.results);
          })
          .catch(function (error) {
            console.error('Search error:', error);
            alert('An error occurred during search. Please try again.');
          });
      });

      // Render the results array into the page
      function renderSearchResults(resultArray) {
        if (!Array.isArray(resultArray) || resultArray.length === 0) {
          const noResults = document.createElement('div');
          noResults.className = 'text-center text-muted';
          noResults.textContent = 'No matching files found.';
          resultsContainer.appendChild(noResults);
          return;
        }

        resultArray.forEach(function (item) {
          const listItem = document.createElement('a');
          listItem.className = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center';
          listItem.href = item.download_url;
          listItem.setAttribute('download', item.filename);

          // File name text
          const nameDiv = document.createElement('div');
          nameDiv.textContent = item.filename;

          // SVG download icon
          const iconSvg = document.createElement('svg');
          iconSvg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
          iconSvg.setAttribute('width', '16');
          iconSvg.setAttribute('height', '16');
          iconSvg.setAttribute('fill', 'currentColor');
          iconSvg.setAttribute('class', 'bi bi-download');
          iconSvg.setAttribute('viewBox', '0 0 16 16');
          iconSvg.innerHTML = `
            <path d="M.5 9.9a.5.5 0 0 1 .5-.5h3v-5a.5.5 
                     0 0 1 1 0v5h3a.5.5 0 0 1 .354.854l-4 4a.5.5 
                     0 0 1-.708 0l-4-4A.5.5 0 0 1 .5 9.9z"/>
            <path d="M2 12.5v1a.5.5 0 0 0 .5.5h11a.5.5 0 0 0 
                     .5-.5v-1a.5.5 0 0 0-1 0v.5H2.5v-.5a.5.5 
                     0 0 0-1 0z"/>
          `;

          listItem.appendChild(nameDiv);
          listItem.appendChild(iconSvg);
          resultsContainer.appendChild(listItem);
        });
      }
    });
  </script>
</body>
</html>
'''
# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────
@application.route('/')
def route_index():
    """
    Serve the HTML page (front end).
    """
    return render_template_string(HTML_TEMPLATE)
@application.route('/api/search_files')
def route_search_files():
    """
    API endpoint:
      - Query parameter: ?query=<search_keyword>
      - Returns JSON:
          {
            "status": 0,
            "results": [
              {"filename": "file1.txt", "download_url": "/download/file1.txt"},
              ...
            ]
          }
    """
    search_keyword = request.args.get('query', '').strip().lower()
    found_files = []
    if search_keyword:
        # Walk through FILES_DIRECTORY
        for current_directory, dirnames, filenames in os.walk(FILES_DIRECTORY):
            for filename in filenames:
                # Case-insensitive substring match
                if search_keyword in filename.lower():
                    # Build relative path for download URL
                    relative_dir = os.path.relpath(current_directory, FILES_DIRECTORY)
                    if relative_dir == '.':
                        relative_dir = ''
                    relative_path = os.path.join(relative_dir, filename).replace('\\', '/')
                    download_url = f"/download/{relative_path}"
                    found_files.append({
                        "filename": filename,
                        "download_url": download_url
                    })

    return jsonify(status=0, results=found_files)
@application.route('/download/<path:relative_path_to_file>')
def route_download_file(relative_path_to_file):
    """
    Serve the requested file as an attachment.
    """
    return send_from_directory(directory=FILES_DIRECTORY,
                               path=relative_path_to_file,
                               as_attachment=True)
# ──────────────────────────────────────────────────────────────────────────────
# APPLICATION ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Ensure the files directory exists
    if not os.path.isdir(FILES_DIRECTORY):
        os.makedirs(FILES_DIRECTORY)
        print(f"Created files directory: {FILES_DIRECTORY}")

    # Start the Flask development server
    application.run(host='0.0.0.0', port=5000, debug=True)
