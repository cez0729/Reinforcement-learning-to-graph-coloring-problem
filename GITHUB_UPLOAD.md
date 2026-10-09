# Upload to a personal GitHub repository

Create an empty repository on GitHub first. Then run these commands from this directory:

```bash
git init
git branch -M main
git add .
git commit -m "Release code and paper sources for RLTabucol to ACO transfer"
git remote add origin https://github.com/<your-account>/<your-repository>.git
git push -u origin main
```

Do not use the old `Corey30` remote from the original project. Replace the placeholder URL with the URL of your own repository.

Before pushing, inspect the staged file list with:

```bash
git status
git diff --cached --stat
```

The release package intentionally excludes formal raw logs and checkpoints. Keep those in a separate private archive if they are needed for later auditing.
