# Kaizen

I jot down things that I wrote down.

## How a post goes live
- Posts on `master` of this repo appear on [thomaspaul.dev](https://www.thomaspaul.dev/) through `mrtpk/mrtpk.github.io` (its `publish` workflow builds this repo as `_posts`).
- The site rebuilds **every Monday at 07:00 IST**. To publish at once: `gh workflow run publish.yml -R mrtpk/mrtpk.github.io`.
- Drafts live in the private `mrtpk/kaizen-drafts` repo. Only what is merged to `master` here becomes public.
