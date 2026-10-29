#!/usr/bin/env bash
#
# Rebase this fork's local commits onto an upstream release tag, resolving
# conflicts with the pi coding agent.
#
# Rewrites the fork's branch as: <base tag> + our commits. The base is the newest
# matching release tag, or the exact tag in UPSTREAM_TAG when that is set. Only
# this job ever writes to the fork; upstream is only ever fetched.
#
# Environment:
#   UPSTREAM_URL       git URL of the repository being forked     (required
#                      unless an `upstream` remote already exists)
#   UPSTREAM_REF       upstream branch to track instead of release tags, e.g.
#                      main. The base becomes that branch's tip on every run, so
#                      the fork follows upstream commit by commit. Use it when
#                      the fork has to match upstream exactly (development
#                      forks, builds consumed at HEAD) rather than ride on
#                      released tags. Overrides the tag settings below.
#   UPSTREAM_TAG       exact release tag to use as the base instead of the
#                      newest one. Set it when something else pins the same tag
#                      - a distro package whose dependency store is hashed, for
#                      instance: a fork that drifts ahead of that pin breaks
#                      that other build.
#   UPSTREAM_TAG_PREFIXES  tag prefixes to track, comma/space separated
#                          (default: v; UPSTREAM_TAG_PREFIX still works)
#   FORK_BRANCH        branch to rewrite                          (default: main)
#   OPENCODE_API_KEY   provider credential for pi  (required only on conflict)
#   PI_PROVIDER        provider for pi             (default: opencode-go)
#   PI_MODEL           model for pi                (default: deepseek-v4.1-flash)
#   PI_MAX_ATTEMPTS    conflict-resolution rounds  (default: 8)
#   DRY_RUN            set to 1 to stop before pushing
#
set -euo pipefail

UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
# Git URL of the repository this fork tracks. Required: the remote is only
# created when it does not already exist.
UPSTREAM_URL="${UPSTREAM_URL:-}"
# Releases, not main. A released tag is a stable base for both building and
# patching: tooling that pins a tag keeps working, and a patch that does not
# touch dependency lockfiles stays applicable to the next release too.
#
# Several prefixes may be given for repositories that release per component
# (browser-v*, desktop-v*, cli-v*). The newest matching tag by commit date wins,
# so the base is whichever line released most recently.
UPSTREAM_TAG_PREFIXES="${UPSTREAM_TAG_PREFIXES:-${UPSTREAM_TAG_PREFIX:-v}}"
# Pin the base to one exact tag. Empty means track the newest matching tag.
UPSTREAM_TAG="${UPSTREAM_TAG:-}"
# Track a branch instead of tags. Non-empty wins over both tag settings.
UPSTREAM_REF="${UPSTREAM_REF:-}"
FORK_BRANCH="${FORK_BRANCH:-main}"
PI_PROVIDER="${PI_PROVIDER:-opencode-go}"
PI_MODEL="${PI_MODEL:-deepseek-v4.1-flash}"
PI_MAX_ATTEMPTS="${PI_MAX_ATTEMPTS:-8}"
DRY_RUN="${DRY_RUN:-0}"
BOT_NAME="${BOT_NAME:-upstream-rebase-bot}"
BOT_EMAIL="${BOT_EMAIL:-upstream-rebase-bot@users.noreply.github.com}"

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "ERROR: $*"; exit 1; }

rebase_in_progress() {
  [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]
}

# The configured prefixes, one per line.
tag_prefixes() {
  printf '%s\n' "$UPSTREAM_TAG_PREFIXES" | tr ', ' '\n' | sed '/^$/d'
}

# Newest matching release tag by commit date. Date rather than version order so
# that per-component release lines compare against each other correctly.
newest_tag() {
  local patterns=()
  while IFS= read -r prefix; do
    patterns+=("refs/tags/${prefix}*")
  done < <(tag_prefixes)
  # sed, not head: head closes the pipe as soon as it has a line, and with a
  # large tag list (hundreds of tags) git dies of SIGPIPE, which pipefail then
  # turns into a failed job. sed consumes the whole stream.
  git for-each-ref --sort=-creatordate --format='%(refname:short)' \
    "${patterns[@]}" | sed -n '1p'
}

conflicted_files() {
  git diff --name-only --diff-filter=U
}

# pi drives git itself, so give it the same non-interactive git behaviour the
# rest of this script relies on (no editor, no pager, no credential prompts).
export GIT_EDITOR=true
export GIT_PAGER=cat
export GIT_TERMINAL_PROMPT=0

git config user.name "$BOT_NAME"
git config user.email "$BOT_EMAIL"
git config rerere.enabled true

if [ -n "$UPSTREAM_REF" ]; then
  log "Fetching branch ${UPSTREAM_REF} from ${UPSTREAM_REMOTE}..."
else
  log "Fetching tags for: $(tag_prefixes | tr '\n' ' ')..."
fi
if ! git remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1; then
  [ -n "$UPSTREAM_URL" ] ||
    die "Remote ${UPSTREAM_REMOTE} does not exist and UPSTREAM_URL is unset."
  git remote add "$UPSTREAM_REMOTE" "$UPSTREAM_URL"
fi
if [ -n "$UPSTREAM_REF" ]; then
  git fetch --no-tags "$UPSTREAM_REMOTE" \
    "+refs/heads/${UPSTREAM_REF}:refs/remotes/${UPSTREAM_REMOTE}/${UPSTREAM_REF}"
else
  while IFS= read -r prefix; do
    git fetch --no-tags "$UPSTREAM_REMOTE" \
      "+refs/tags/${prefix}*:refs/tags/${prefix}*"
  done < <(tag_prefixes)
  # A pinned tag need not match any prefix, so fetch it by name too.
  if [ -n "$UPSTREAM_TAG" ]; then
    git fetch --no-tags "$UPSTREAM_REMOTE" \
      "+refs/tags/${UPSTREAM_TAG}:refs/tags/${UPSTREAM_TAG}"
  fi
fi
git fetch --no-tags origin "$FORK_BRANCH"

if [ -n "$UPSTREAM_REF" ]; then
  TARGET_TAG="${UPSTREAM_REMOTE}/${UPSTREAM_REF}"
  UPSTREAM_SHA="$(git rev-parse "${TARGET_TAG}^{commit}")" ||
    die "Upstream branch ${UPSTREAM_REF} not found on ${UPSTREAM_REMOTE}."
  log "Base is ${UPSTREAM_REF} (${UPSTREAM_SHA:0:12})"
elif [ -n "$UPSTREAM_TAG" ]; then
  TARGET_TAG="$UPSTREAM_TAG"
  git rev-parse --verify --quiet "refs/tags/${TARGET_TAG}^{commit}" >/dev/null ||
    die "Pinned UPSTREAM_TAG ${TARGET_TAG} not found on ${UPSTREAM_REMOTE}."
  UPSTREAM_SHA="$(git rev-parse "${TARGET_TAG}^{commit}")"
  log "Base is ${TARGET_TAG} (${UPSTREAM_SHA:0:12})"
else
  TARGET_TAG="$(newest_tag)"
  [ -n "$TARGET_TAG" ] ||
    die "No tag matching $(tag_prefixes | tr '\n' ' ') found on ${UPSTREAM_REMOTE}."
  log "Newest release is ${TARGET_TAG}"
  UPSTREAM_SHA="$(git rev-parse "${TARGET_TAG}^{commit}")"
  log "Base is ${TARGET_TAG} (${UPSTREAM_SHA:0:12})"
fi

git checkout -B "$FORK_BRANCH" "origin/$FORK_BRANCH"

if git merge-base --is-ancestor "$UPSTREAM_SHA" HEAD; then
  log "Already up to date with upstream; nothing to rebase."
  exit 0
fi

OUR_COMMITS="$(git rev-list --count "$UPSTREAM_SHA"..HEAD)"
log "Rebasing ${OUR_COMMITS} local commit(s) onto upstream ..."

if git rebase "$UPSTREAM_SHA"; then
  log "Rebase completed cleanly."
else
  # The checkout leaves push credentials in .git/config, and pi runs with bash.
  # Point the push URL somewhere unroutable for the agent phase, so a resolution
  # can never publish: only this script pushes, after the agent has finished.
  origin_push_url="$(git remote get-url --push origin)"
  git remote set-url --push origin "https://push-disabled.invalid/"
  restore_push_url() { git remote set-url --push origin "$origin_push_url"; }

  command -v pi >/dev/null 2>&1 || {
    git rebase --abort || true
    die "Rebase hit conflicts but pi is not installed; aborting."
  }
  [ -n "${OPENCODE_API_KEY:-}" ] || {
    git rebase --abort || true
    die "Rebase hit conflicts but OPENCODE_API_KEY is unset; aborting."
  }

  attempt=0
  while rebase_in_progress; do
    attempt=$((attempt + 1))
    if [ "$attempt" -gt "$PI_MAX_ATTEMPTS" ]; then
      git rebase --abort || true
      die "pi did not converge after ${PI_MAX_ATTEMPTS} attempts; aborted."
    fi

    files="$(conflicted_files)"
    [ -n "$files" ] || die "Rebase is in progress with no conflicted files; giving up."

    log "Conflict round ${attempt}: $(echo "$files" | tr '\n' ' ')"
    printf '%s\n' "$files" > /tmp/conflicted-files.txt

    # --no-mcp/-extensions/-skills keep CI deterministic: the runner has none of
    # the user's interactive agent configuration. --approve lets pi read the
    # repository's own AGENTS.md, which is the relevant guidance here.
    pi --print \
      --provider "$PI_PROVIDER" \
      --model "$PI_MODEL" \
      --api-key "$OPENCODE_API_KEY" \
      --no-mcp --no-extensions --no-skills \
      --approve \
      --append-system-prompt "$(cat <<'PROMPT'
You are resolving a git rebase conflict inside a CI job. There is no human to
ask: act autonomously and never wait for input.

Rules:
- The checkout is mid-rebase. Replay our local commits onto upstream's code.
  Our commits are small, local patches; upstream's version is the new baseline.
- Preserve our patches' intent. When upstream already implements or supersedes
  one of our patches, keep upstream's version and drop our now-redundant hunk.
- Never resolve by deleting whole files, discarding our commits, or taking
  "ours"/"theirs" wholesale. Edit the conflicting regions.
- Keep the result compiling and formatted the way the repo expects.
- Never run `git rebase --abort`, `git reset --hard`, or `git checkout --force`.
- When the working tree for the current commit is correct, stage the files and
  run `GIT_EDITOR=true git rebase --continue`. Repeat until the rebase finishes.
- Finish by printing a one-paragraph summary of how you resolved each file.
PROMPT
)" \
      "Resolve the git rebase conflicts for this commit. The conflicted files are listed in /tmp/conflicted-files.txt. Any commit that still has to be replayed afterwards must be handled the same way."

    if [ -n "$(conflicted_files)" ]; then
      die "pi exited with conflicts still unresolved in: $(conflicted_files | tr '\n' ' ')"
    fi

    if rebase_in_progress && ! GIT_EDITOR=true git rebase --continue; then
      # A staged commit that no longer changes anything (upstream already has
      # the patch) makes --continue refuse with "No changes". Drop that commit.
      if [ -z "$(git diff --cached --name-only)" ]; then
        log "Commit became empty after resolution; skipping it."
        GIT_EDITOR=true git rebase --skip || true
      fi
    fi
  done
  log "Rebase completed with pi assistance after ${attempt} round(s)."
fi

# Only meaningful when tracking tags: a pinned base stays the base whatever
# releases, and a branch base is simply the branch as of this run - moving on
# while we rebase is what the next run is for.
if [ -z "$UPSTREAM_REF" ] && [ -z "$UPSTREAM_TAG" ] && [ "$(newest_tag)" != "$TARGET_TAG" ]; then
  die "A newer release tag appeared while we were rebasing; leaving the fork untouched for the next run."
fi

if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN=1: rebase result left in place, not pushing."
  git --no-pager log --oneline -n "$((OUR_COMMITS + 3))"
  exit 0
fi

# Only defined when the agent phase ran and disabled the push URL.
if declare -F restore_push_url >/dev/null; then
  restore_push_url
fi
log "Local commits after rebase:"
git --no-pager log --oneline "$UPSTREAM_SHA"..HEAD

# This branch is only ever advanced by this job, so force-with-lease against the
# exact ref we fetched above is the safe way to publish a rewritten history.
ORIGIN_SHA="$(git rev-parse "origin/${FORK_BRANCH}")"
git push \
  --force-with-lease="refs/heads/${FORK_BRANCH}:${ORIGIN_SHA}" \
  origin "HEAD:refs/heads/${FORK_BRANCH}"
log "Pushed ${FORK_BRANCH} to origin."
