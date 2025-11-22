What ci.yml does?

CI Pipeline: .github/workflows/ci.yml : This pipeline is triggered on every push or PR to main or dev.
What it does:
- Runs on PRs and pushes to main/dev
- Installs dependencies, runs linting and tests
- Checks that the version is unique (to prevent accidental duplicate releases)

CD Pipeline: .github/workflows/cd.yml
This pipeline is triggered after a successful merge to main.

What it does:
- Only runs on push to main
- Builds the wheel (databricks bundle deploy takes care of that)
- Deploys the Lakeflow job to both acceptance and production using environment-specific secrets
- Uses the Databricks CLI to deploy bundles

Here's a continuation of your README:

---

## CI/CD Pipelines

### CI Pipeline: `.github/workflows/ci.yml`

This pipeline runs automated checks on every pull request to `main`.

**Trigger:** Pull requests to `main` branch

**What it does:**

1. **Checkout code** - Fetches the full git history of the branch
2. **Create git tag from version** - Reads `version.txt` and creates a temporary git tag (for validation purposes)
3. **Install uv** - Sets up the uv package manager
4. **Install dependencies** - Runs `uv sync --extra test` to install project dependencies including test extras
5. **Run pre-commit checks** - Executes all pre-commit hooks (linting, formatting, type checking, etc.)
6. **Run pytest** - Runs all tests except those marked with `ci_exclude`

**Purpose:** Ensures code quality and that all tests pass before merging to `main`.

---

### CD Pipeline: `.github/workflows/cd.yml`

This pipeline deploys your Databricks bundles to acceptance and production environments.

**Triggers:** 
- Push to `main` branch (automatic)
- Manual trigger via `workflow_dispatch`

**What it does:**

1. **Matrix deployment** - Runs deployment twice in parallel: once for `acc` (acceptance) and once for `prd` (production)
2. **Checkout code** - Fetches the repository code
3. **Install Databricks CLI** - Sets up version 0.246.0 of the Databricks CLI
4. **Configure authentication** - Creates a `~/.databrickscfg` file with the `marvelous` profile using:
   - `DATABRICKS_HOST` from GitHub environment variables
   - `DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET` from GitHub secrets (service principal credentials)
5. **Install uv** - Sets up the uv package manager
6. **Deploy to Databricks** - Runs `databricks bundle deploy` for each environment:
   - Passes current git SHA and branch name as variables
   - Uses environment-specific configurations from `databricks.yml`
   - For `prd` only: Creates and pushes a git tag based on `version.txt`

**Environment-specific secrets:**
Each environment (`acc`, `prd`) must have these configured in GitHub:
- `DATABRICKS_HOST` (variable)
- `DATABRICKS_CLIENT_ID` (secret)
- `DATABRICKS_CLIENT_SECRET` (secret)

**Purpose:** Automates deployment of your ML pipeline to Databricks, ensuring consistent deployments with proper versioning and traceability.





## Deploy to Databricks - Line by Line

### 1. Set Environment Variable

```yaml
- name: Deploy to Databricks
  env:
    DATABRICKS_BUNDLE_ENV: ${{ matrix.environment }}
```

Sets environment variable to either `"acc"` or `"prd"`. Tells Databricks CLI which target from `databricks.yml` to use.

---

### 2. Deploy Bundle Command

```bash
databricks bundle deploy \
```

Deploys your bundle (wheel package + jobs) to Databricks workspace. Uses the configuration from `databricks.yml` for the current environment.

---

### 3. Pass Git SHA Variable

```bash
  --var="git_sha=${{ github.sha }}" \
```

Passes the current commit SHA (e.g., `"a1b2c3d4..."`) as a variable.
- `github.sha` = unique ID of the commit that triggered this workflow
- Used for tracking which code version is deployed

---

### 4. Pass Branch Name Variable

```bash
  --var="branch=${{ github.ref_name }}"
```

Passes the branch name (e.g., `"main"`) as a variable.
- `github.ref_name` = name of the branch that triggered the workflow
- These variables can be used in your `databricks.yml` (e.g., in job names or tags)

---

### 5. Production-Only: Create Git Tag

```bash
if [ "${{ matrix.environment }}" = "prd" ]; then
```

Checks if currently deploying to production. Only runs the following commands for `prd`, not `acc`.

```bash
  echo "VERSION=$(cat version.txt)"
```

Reads `version.txt` file and prints it (e.g., `"VERSION=1.2.3"`). Just for logging/debugging purposes.

```bash
  git tag $VERSION
```

Creates a git tag with the version number (e.g., tag `"1.2.3"`). Marks this commit as a release in git history.

```bash
  git push origin $VERSION
```

Pushes the tag to the remote repository (GitHub). Makes the release tag visible in GitHub's releases/tags page.

```bash
fi
```

Ends the if statement.

---

**Summary:** Deploys to Databricks with tracking info (commit SHA + branch), then for production only, creates and pushes a git tag for versioning.