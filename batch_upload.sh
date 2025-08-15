#!/bin/bash

# 1. Go to your repo root
cd "/c/Users/mcjca/Documents/Microbiome/Git/Interacting-Hosts-with-Microbe-Exchange" || exit

# 2. Make sure branch is linked to GitHub remote
git branch -M main
git push --set-upstream origin main || true

# 3. Reset staging area and clear any partial adds
git reset
git restore --staged .

# 4. Find all files except .git, and batch them
batch_size=100
count=0
batch=()

# Recursively list files, excluding .git directory
while IFS= read -r file; do
    batch+=("$file")
    ((count++))
    
    if ((count == batch_size)); then
        echo "Committing batch of $batch_size files..."
        git add "${batch[@]}"
        git commit -m "Add next $batch_size files"
        git push
        batch=()
        count=0
    fi
done < <(find . -type f ! -path "./.git/*")

# Commit any leftover files
if ((count > 0)); then
    echo "Committing final batch of $count files..."
    git add "${batch[@]}"
    git commit -m "Add final $count files"
    git push
fi

echo "✅ All files uploaded in batches."
