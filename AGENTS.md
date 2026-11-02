@RTK.md

# Coding Principles

## Principles Adapted from the Zen of Python

Favor elegant solutions over untidy ones.

State intent directly instead of leaving it implied.

Choose simplicity over complexity.

When complexity is necessary, keep it structured rather than convoluted.

Prefer flat structures to deeply nested ones.

Give code room to breathe instead of packing it tightly.

Make readability a priority.

Do not abandon established rules merely to accommodate a special case.

Let practical needs take precedence over theoretical purity.

Make errors visible.

Suppress them only through an explicit decision.

When something is ambiguous, do not guess.

Aim for one clear way to accomplish a task, ideally without competing obvious alternatives.

That approach may take time to recognize, unless you happen to be Dutch.

Taking action now is preferable to never acting.

Still, leaving something undone can be better than rushing it immediately.

Treat an implementation that is difficult to explain as a poor design.

A design that is easy to explain may be a sound choice.

Use namespaces freely to keep names organized.

## Design and Maintenance

Write for the next reader, whether human or agent. Code produced by one agent often becomes context for another. Agent involvement does not reduce the need for clear organization, readable code, or useful documentation. Concise code that is easy to understand allows more of the codebase to fit within a reader's available context.

Keep simplifying. Use a straightforward solution whenever one is available, without sacrificing performance. Add complexity only when its benefits justify its cost.

Remove AI-style filler. Avoid ornate language and elaborate implementations that serve no practical purpose.

Prevent tangled code by dividing behavior into clear functions, modules, and other components. Define their inputs and outputs so internal logic can change without introducing hidden dependencies. These boundaries should make changes easier to manage.

Test behavior at meaningful component boundaries rather than exhaustively checking every function or line. Express cases through inputs and expected outputs, and verify that behavior remains stable. Where appropriate, share a test function across multiple cases and combine related checks. Read existing tests before adding new ones, follow their style and approach, and seek broad behavioral coverage with little test code.

Keep code and its documentation self-contained. Refer outside the documented function, class, or module only when that reference is useful in the specific context.

## Implementation and Documentation

Document public APIs, classes, and functions with docstrings when their names and signatures do not make their behavior, constraints, or side effects clear.

Use docstrings to add useful information rather than repeat the implementation.

Add inline comments when they make the code easier to read.

Write text in ASD-STE100 Simplified Technical English to support clear understanding.

Keep historical explanations and references to earlier implementations out of docstrings and inline comments.

Favor class-based implementations over function-based alternatives.
