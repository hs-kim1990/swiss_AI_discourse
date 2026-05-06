@echo off
echo ================================================
echo  Swiss AI Discourse — Push Dashboard to GitHub
echo ================================================
echo.

cd /d "%~dp0"

echo [1/4] Pruning stale worktree references...
git worktree prune
if errorlevel 1 (
    echo Warning: worktree prune had issues, continuing...
)

echo.
echo [2/4] Staging docs/ folder...
git add docs/
if errorlevel 1 ( echo ERROR: git add failed & pause & exit /b 1 )

echo.
echo [3/4] Committing...
git commit -m "Add dashboard (docs/) — 10M Initiative visualisation"
if errorlevel 1 ( echo ERROR: git commit failed & pause & exit /b 1 )

echo.
echo [4/4] Pushing to origin/main...
git push origin main
if errorlevel 1 ( echo ERROR: git push failed & pause & exit /b 1 )

echo.
echo ================================================
echo  Done! Now enable GitHub Pages:
echo.
echo  1. Go to: https://github.com/hs-kim1990/swiss_AI_discourse
echo  2. Settings - Pages
echo  3. Source: Deploy from a branch
echo  4. Branch: main   Folder: /docs
echo  5. Save
echo.
echo  Your site will be live in ~60 seconds at:
echo  https://hs-kim1990.github.io/swiss_AI_discourse/
echo ================================================
pause
