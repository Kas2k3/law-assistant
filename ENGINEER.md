# Engineering Standard AI Team v1.0

> **Standard source for all technical activities of AI Team V365**
>
> This document defines shared standards for repo, code, git, pr, test, ci, release, docs,...
>
> Each repository may add a `PROJECT-GUIDELINES.md` file for project-specific rules or exceptions, however project-level rules must not conflict with this document.

---

## 1. Purpose

The purpose of this document is to create a unified, maintainable and professional technical process to make maintenance easier and ensure the quality of current projects is consistent.

This standard aims to:

- make repositories easy to understand and maintain
- keep code clean, readable and consistent
- standardize collaboration and code review
- reduce errors that can be detected early before merge or deploy
- ensure technical decisions and releases can be traced
- build a shared engineering culture across the whole team

---

## 2. Scope

This standard applies to:

- source code repositories
- backend, frontend, AI, data, infrastructure and automation projects,..
- internal tools and production systems
- proof of concept projects that continue to be maintained by the team
- all contributors working in AI Team repositories

Experiments may use a slightly more flexible process during the research phase. However, if code is merged into shared branches it must still comply with the general standard when merged

## 3. Repository Standard

Every repository must:

- be easy to understand
- be logically organized
- not contain unnecessary files
- be appropriate to the responsibilities of the project
- provide enough information for another person to run and maintain the project

repository must follow these principles:

- related code should be grouped together
- responsibilities between modules should be clear
- source code, test, config, script, asset and documentation should not be mixed without reason
- temporary files, generated files, local environments, secrets and build artifacts must not be committed unless actually necessary
- avoid overly deep folder structures if they do not make the code clearer
- do not use unclear folder names such as `new`, `final`, `test2`, `temp`, `backup` for the long term
- the repository root should be clean and contain only project-level files and main directories.

A repository should provide enough information to answer:

- what is this project
- how does the system work at a high level
- how to install dependencies
- how to configure the environment
- how to run locally
- how to run tests
- how to build
- how to deploy if applicable.

The project may use files or directories:

```text
README.md
PROJECT-GUIDELINES.md
CONTRIBUTING.md
SECURITY.md
CHANGELOG.md
docs/
tests/
scripts/
```

Not every repository is required to have all of the files above. Only add them when they actually ship value

---

## 4. Git Commit Standard

All commits must follow the format:

```text
<type>: <description>
```

Example:

```text
feat: add user authentication
fix: handle empty api response
refactor: simplify document parser
docs: update local setup guide
test: add authentication tests
chore: update dependencies
```

### 5.1 Standard Commit Type

| Type       | Meaning                                                  |
| ---------- | -------------------------------------------------------- |
| `feat`     | add a new feature or capability                          |
| `fix`      | fix a bug or incorrect behavior                          |
| `refactor` | reorganize code without changing intended behavior       |
| `perf`     | improve performance                                      |
| `docs`     | change documentation only                                |
| `test`     | add or update tests                                      |
| `style`    | change format style, without changing logic              |
| `build`    | changes related to build system or dependencies          |
| `ci`       | change CI CD workflow                                    |
| `chore`    | maintenance that does not belong to another group        |
| `revert`   | revert a previous change                                 |

### 5.2 Commit Writing Rules

Commit message must:

- be written completely in English
- be lowercase
- have a short description
- clearly describe the action performed
- each commit only represents one logical change
- use normal plain text
- do not use unnecessary punctuation or special characters in the description.

Should write:

```text
feat: add roadmap generation
fix: prevent duplicate records
refactor: simplify retrieval pipeline
```

Should not write:

```text
update
fix bug
final
final2
test
changes api
feat: Add Login
fix: fixed some things
```

A good commit should answer:

> What logical change does this commit create?

Large and unrelated changes should be split into separate commits.

---

## 6. Branch Naming Standard

Branch must follow the format:

```text
<type>/<ticket-description>
```

Example:

```text
feature/142-user-authentication
fix/218-upload-timeout
refactor/305-retrieval-pipeline
docs/112-api-guide
test/078-authentication-flow
chore/091-update-dependencies
```

If the project does not use a ticket ID:

```text
feature/user-authentication
fix/upload-timeout
refactor/retrieval-pipeline
```

### 6.1 Standard Branch Type

Preferred branch types:

```text
feature
fix
refactor
docs
test
chore
hotfix
```

Branch name must:

- be written in English
- be lowercase
- be short but easy to understand
- use hyphens `-` between words
- describe one main change.

Should not use:

```text
vuong-test
new
feature1
final
test123
fix-code
```

---

## 7. Branch Strategy

Default workflow:

```text
feature/* or fix/*
        |
        v
       dev
        |
        v
       main
        |
        v
   production
```

### 7.1 Branch `dev`

`dev` is the shared integration branch.

Normal development code must be merged into `dev` through Pull Request.

Direct push to `dev` will be restricted.

### 7.2 Branch `main`

`main` represents production ready code.

All changes going into `main` must go through Pull Request.

Direct push to `main` must be completely blocked.

### 7.3 Production

Production deployment must originate from `main` or from an approved release mechanism based on `main`.

### 7.4 Hotfix

Urgent production fixes may use:

```text
hotfix/<ticket-description>
```

Hotfixes will still need review and approval before merge. After a production hotfix, the corresponding change needs to exist in `dev`

---

## 8. Pull Request Standard

Pull Request is the standard unit for code review and merge approval

PR title must follow the same format as commit:

```text
<type>: <description>
```

Example:

```text
feat: add google authentication
fix: handle document upload timeout
refactor: simplify readiness scoring
```

PR title must:

- be written in English
- be lowercase
- be short and clear
- describe the main logical change.

### 8.1 Pull Request Content

Pull Request should have the structure:

```markdown
## what

what does this pull request change

## why

why is this change needed

## changes

- main change
- related change

## testing

how was this change tested

## risks

known risks or side effects

## screenshots

add screenshots when the change affects the user interface

## checklist

- [ ] code is clean and reviewed by the author
- [ ] tests are added or updated when needed
- [ ] ci checks pass
- [ ] no secrets or sensitive data are committed
- [ ] documentation is updated when needed
```

PR description should be concise.

The purpose is to help the reviewer understand the change, not to describe the entire implementation again.

### 8.2 Pull Request Size

Pull Request should focus on one logical change.

Avoid combining:

- multiple unrelated features
- large refactor and new feature in the same PR
- dependency upgrade with unrelated feature
- formatting the entire repository at the same time as business logic change.

Small and focused PRs make review, test and revert easier.

---

## 9. Pull Request Review Standard

Every Pull Request will be approved before merge. Dev must NOT merge a Pull Request without approval.

Review focuses on:

- correctness
- maintainability
- architecture
- code clarity
- security
- tests for important behavior
- backward compatibility
- operational risk.

PR author should self-review their diff before requesting review.

Important review conversations should be resolved before merge.

If there are significant changes after the PR has been approved, another review will be requested

---

## 10. Code Standard

Code standards need to ensure clean code

### 10.1 Language

Source code must use English for:

- variable name
- function name
- class name
- file and module name where appropriate
- code comment
- error identifier
- technical constant and enum.

User facing content may use the language required by the product.

### 10.2 Naming

Names must be meaningful and clearly express intent.

Should use:

```text
calculate_total_cost
document_parser
request_timeout
user_repository
```

Should not use:

```text
x
tmp2
abc
func1
data_new
final_result2
```

Short names may be used for common concepts in a local scope if the context is clear enough.

### 10.3 Clean Code

Code must prioritize readability and maintainability.

Dev should:

- keep functions focused on one responsibility
- keep modules focused on one clear domain or responsibility
- reduce unnecessary complexity
- remove unused code
- avoid duplicate logic when a reasonable abstraction can be created
- avoid premature abstraction
- avoid deeply nested logic if simpler control flow can be written
- express important business rules clearly
- keep interfaces small and easy to understand
- prefer simple solutions over smart but difficult-to-read solutions.

### 10.4 Code Organization

Code should be organized by clear responsibility.

A project should help developers easily identify:

- entry point
- business logic
- external integration
- configuration
- data access
- shared utility
- test.

Large files should be split when they contain multiple unrelated responsibilities.

Reusable logic should be placed in an appropriate module instead of copied in multiple places.

### 10.5 Function and Method

Function should:

- perform one clear task
- have easy-to-understand input and output
- avoid hidden side effects
- keep a reasonable size
- use names that clearly describe intent.

A function with too many arguments should be reconsidered if those arguments actually represent one common object or configuration.

### 10.6 Comment

Comment must be written in English, concise and used to explain:

- why this decision exists
- why this special behavior is necessary
- important constraints
- non-obvious business logic.

Avoid comments that only repeat the code.

Should:

```text
retry once because the upstream service occasionally returns a transient timeout
```

Should not:

```text
increment i by one
```

Commented out code should be removed from the repository.

### 10.7 Error Handling

Error must not be silently ignored.

Code should:

- fail clearly when recovery is not possible
- return meaningful errors
- not expose internal sensitive information to users
- log enough context for debugging
- handle expected failure cases clearly.

### 10.8 Configuration

Environment specific value must NOT be hard-coded into business logic.

Configuration should be separated using a mechanism appropriate for the project.

Secret must NOT be stored directly in source code.

### 10.9 Dead Code

Unused functions, imports, variables, old features and obsolete code should be removed.

Do not keep dead code just because it may be used later.

Git history is where old implementations are stored.

---

## 11. Testing Standard

Important behavior must be testable.

Testing strategy may vary by project, but the team should consider:

- unit test for business logic
- integration test for system boundary
- end-to-end test for critical user flow
- regression test for important bugs.

When fixing an important bug, developer should add a regression test if appropriate.

Test should:

- be deterministic
- be easy to read
- prioritize testing behavior instead of implementation detail
- avoid unnecessary external dependency
- have meaningful test names.

High coverage is not considered the only evidence of software quality.

The goal of testing is to create confidence in important behavior.

---

## 12. Documentation Standard

Documentation needs to be useful, up to date and concise. Each repo needs to provide enough information so that another person can understand and run the project

Documentation should be updated when changes affect:

- setup
- architecture
- API
- configuration
- deployment
- operational procedure
- important user workflow.

---

## 13. PROJECT-GUIDELINES.md

Each repository may have:

```text
PROJECT-GUIDELINES.md
```

This file is used to define project-specific technical rules.

Example:

- project architecture
- framework-specific convention
- required local tool
- testing requirement
- project-specific branch or release behavior
- deployment constraint
- domain-specific terminology
- AI model or dataset requirement.

---

## 14. Security Standard

All contributors must follow basic secure engineering.

- do not commit password, API key, token, private key or production credential
- do not commit `.env` containing secrets
- do not expose sensitive data in logs
- grant only the minimum necessary permissions
- validate untrusted input
- carefully review changes related to authentication and authorization
- keep dependencies in a reasonable state and not too outdated
- immediately report accidentally committed secrets.

If a secret has been committed, deleting the secret line in a later commit is not enough. That secret must be considered compromised and must be rotated.

---

## 15. Dependency Standard

Dependency should be added intentionally.

Before adding a new dependency, consider:

- whether it is actually necessary
- maintenance status
- security risk
- license compatibility
- size and complexity
- whether that capability already exists in current dependencies.

Dependency change should be separated from unrelated features when appropriate. Lock file should be committed if the project ecosystem uses it. Unused dependency must be removed.

---

## 16. Environment Standard

Project should separate environments clearly.

Common environments:

```text
local
development
production
```

Project may also have:

```text
test
staging
demo
```

Do not create multiple codebases only to handle environments.

---

## 17. CI Standard

Dev is responsible for ensuring PR passes the required automated checks.

Depending on the project, CI will perform appropriate checks, for example:

```text
format
lint
type check
test
build
security
dependency validation
```

the general principle is:

> required ci checks must pass before merge

Failing checks must not be bypassed, if CI fails, the author is responsible for understanding and fixing the issue or explaining why that check is invalid.

---

## 18. Release Standard

Code merged into `main` must be considered production ready, unless the project has another approved release mechanism.

Before release, need to confirm:

- required checks have passed
- approval has been completed
- critical behavior has been tested
- configuration is correct
- breaking changes have been identified
- important systems can be rolled back.

Dev does not deploy unreviewed code to production.

---

## 19. Logging and Observability

Production system should provide enough visibility to diagnose errors.

When appropriate, the application should have:

- structured log
- meaningful error message
- request or correlation identifier
- health check
- metric
- tracing
- alerting.

Log should contain enough context for debugging but absolutely must not expose secrets or sensitive data.

---

## 20. AI Engineering Standard

AI projects need additional discipline in reproducibility and evaluation.

When appropriate, the project should track the following information:

- model name and version
- prompt version
- embedding model
- retrieval configuration
- chunking configuration
- dataset version
- evaluation dataset
- evaluation metric
- model parameter
- important preprocessing configuration.

AI changes should not be evaluated only by subjective manual testing if quantitative measurement is possible. Major changes related to model, prompt, retrieval or dataset should be compared with the agreed evaluation baseline.

allow identification of which configuration produced a specific result. For important releases, need to be able to identify:

```text
code version
model version
prompt version
dataset version
evaluation result
```

---

## 21. Definition of Done

```text
implementation completed
+
code is clean
+
tests completed when required
+
required ci checks pass
+
pull request reviewed
+
approved
+
documentation updated when required
+
change merged to the correct branch
```

Task must not be marked Done just because the code runs on the author's machine.

---

## 22. Daily Development Workflow

Default workflow:

```text
1. create or receive a task
2. create a branch from the appropriate base branch
3. implement one logical change
4. keep code clean and focused
5. test locally
6. create clear commits
7. push branch
8. open pull request
9. fix ci errors
10. address review comments
11. receive team lead approval
12. merge into dev
13. verify changes after integration
14. promote approved changes from dev to main
15. deploy main to production
```

---

## 23. Core Engineering Principles

Every member should remember:

1. write code for humans to read first, machines to run second
2. keep each change small and focused on one logical change
3. use English consistently in engineering artifacts
4. keep repositories clean and easy to understand
5. prefer simple solutions
6. automate repetitive quality checks
7. always review before merge
8. do not sacrifice security for convenience
9. test important behaviors
10. document decisions that future engineers will need to know
11. do not merge without Team Lead approval
12. improve the standard when the team finds a better way.

---

# Engineering Standard AI Team v1.0

This document is the **source of truth for all engineering practices of AI Team V365**. Every repository follows this standard by default. Project-specific requirements or exceptions are placed in:

```text
PROJECT-GUIDELINES.md
```
