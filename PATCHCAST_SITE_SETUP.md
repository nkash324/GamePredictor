# PatchCast website

The PatchCastSite folder (next to this one) is the public website. It only contains the finished page and its data, never the
API key, database or code. `publish_site.bat` (in the GamePredictor folder) rebuilds it and
uploads it.

Files:

- `index.html`: the page (copied from `GamePredictor\site\index.html`, so edit it there)
- `data.js`: the latest forecast and accuracy numbers, written by `build_site.py`
- `.nojekyll`: tells GitHub Pages to serve the files as they are

## One-time setup (free, about 5 minutes)

1. On github.com, click **New repository**. Name it `patchcast`, set it to **Public**, and
   leave "Add a README" unticked. Click **Create repository**.
2. Open a terminal in the PatchCastSite folder (in File Explorer, click the address bar, type `cmd`, press
   Enter) and run these, replacing `YOUR-NAME` with your GitHub username:

   ```
   git init
   git branch -M main
   git remote add origin https://github.com/YOUR-NAME/patchcast.git
   git add .
   git commit -m "First version"
   git push -u origin main
   ```

3. On the repository page, go to **Settings → Pages**. Under "Build and deployment", set
   Source to **Deploy from a branch**, Branch to **main** and folder to **/ (root)**, then click
   **Save**.
4. After a minute or two the site is live at `https://YOUR-NAME.github.io/patchcast/`.
   Use that link in the production key application.

## Updating it

Run `update_model.bat`, then double-click `publish_site.bat`. Done.

## Rules this site follows (Riot developer policies)

- No Riot logos and no claim of being endorsed or approved; the legal line is on the About tab.
- Free, no ads, no paid features.
- Champion-level statistics only: no player names, ranks or ratings.
- Doesn't call the Riot API and has no API key in it. Champion pictures load from Data Dragon.
