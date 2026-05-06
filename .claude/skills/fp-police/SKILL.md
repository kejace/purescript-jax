---
name: fp-police
description: Audit a PureScript codebase for functional-programming violations and non-idiomatic shortcuts (unsafeCoerce, unsafePerformEffect, Effect.Ref outside boundaries, partial functions, console.log in prod, error strings instead of ADTs, FFI without EffectFn / Impl-suffix, internal-module imports, etc.). Use when the user asks to "audit", "review for FP violations", "look for shortcuts", or runs "/fp-police". Produces a structured report grouped by VIOLATIONS / WARNINGS / STYLE / INFO.
---

# FP Police: PureScript Code Quality Audit

Audit a PureScript codebase for functional programming violations, shortcuts, and non-idiomatic code that may have accumulated over time.

## Usage

```
/fp-police              # Full audit of current project
/fp-police quick        # Quick scan (unsafe ops + code smells only)
/fp-police <path>       # Audit specific directory
/fp-police ffi          # FFI audit only
/fp-police unsafe       # Unsafe operations audit only
/fp-police style        # Idiom/style audit only
```

## Arguments

$ARGUMENTS

## Why This Exists

Claude is highly motivated to solve problems, which sometimes means taking shortcuts that create technical debt — using `unsafeCoerce` to avoid a tricky type, reaching for `Effect.Ref` instead of `StateT`, writing a string where an ADT belongs. This audit catches what slipped through.

## Instructions

You are the FP Police auditor. Use the Grep tool for all searches (never bash grep). Run the appropriate categories based on arguments, then generate a structured report.

### 1. Parse the Audit Scope

- No arguments or "full": Run all categories
- "quick": Categories A + B only
- Specific path: Scope all searches to that directory
- "ffi": Category C only
- "unsafe": Category A only
- "style": Category D only

### 2. Run the Audit

#### Category A: Unsafe Operations

**A1. unsafeCoerce (VIOLATION)**
```
Pattern: "unsafeCoerce"
Glob: **/*.purs
```
Almost never justified in application code. Acceptable in library internals with documented safety proof.

**A2. unsafePerformEffect (VIOLATION)**
```
Pattern: "unsafePerformEffect"
Glob: **/*.purs
```
This is a time bomb. The compiler may inline, reorder, or eliminate the call.

**A3. unsafePartial (WARNING)**
```
Pattern: "unsafePartial"
Glob: **/*.purs
```
Indicates incomplete pattern matching. Review if the partiality is genuinely safe.

**A4. unsafeCrashWith (WARNING)**
```
Pattern: "unsafeCrashWith"
Glob: **/*.purs
```
Acceptable for genuinely unreachable code. Should be documented.

**A5. Effect.Ref in non-boundary code (WARNING)**
```
Pattern: "Effect\.Ref|import.*\bRef\b"
Glob: **/*.purs
```
Ref is for shared mutable state at application boundaries. For local accumulation, use StateT, foldl, or ST.

#### Category B: Code Smells

**B1. Console.log in production code (WARNING)**
```
Pattern: "Console\.log|import.*Console"
Glob: **/*.purs (exclude test/)
```

**B2. Partial functions (WARNING)**
```
Pattern: "fromJust|unsafeIndex"
Glob: **/*.purs
```

**B3. else pure unit (STYLE)**
```
Pattern: "else pure unit|else\s+pure\s+unit"
Glob: **/*.purs
```
Should be `when` or `unless`.

**B4. Show used for serialization (WARNING)**
```
Pattern: "show.*writeFile|show.*stringify|show.*encode|writeFile.*show"
Glob: **/*.purs
```
`Show` is for debugging. Use a codec for data that crosses boundaries.

**B5. TODO/FIXME/HACK markers (INFO)**
```
Pattern: "TODO|FIXME|HACK|XXX|DEPRECATED"
Glob: **/*.purs
```
Review for actionability.

**B6. Error strings instead of ADTs (WARNING)**
```
Pattern: 'Left ".*"|throwError ".*"'
Glob: **/*.purs
```
String errors lose structure. Model domain errors as ADTs.

#### Category C: FFI Discipline

**C1. FFI files in unexpected places (REVIEW)**
```
Glob: src/**/*.js
```
Each FFI file should be minimal and have a corresponding .purs wrapper.

**C2. Foreign imports without Impl suffix (STYLE)**
```
Pattern: "foreign import (?!.*Impl)"
Glob: **/*.purs
```
Convention: suffix foreign imports with `Impl` and hide behind a PureScript wrapper.

**C3. Curried foreign imports (STYLE)**
```
Look for `foreign import` lines with multiple `->` that don't use EffectFn/Fn types.
Pattern: "foreign import.*->.*->.*->"
Glob: **/*.purs
```
Should use `EffectFn`/`Fn` for uncurried FFI.

**C4. Point-free runFn (WARNING)**
```
Pattern: "= runFn[0-9]+ \w+$"
Glob: **/*.purs
```
`runFn` must be fully saturated for inlining. Write `foo a b = runFn2 impl a b`.

#### Category D: Style & Idioms

**D1. Internal module imports (VIOLATION)**
```
Pattern: "import.*\.Internal\."
Glob: **/*.purs (exclude the defining library's own source)
```
Application code should use public APIs.

**D2. Missing type signatures (WARNING)**
```
Look for top-level bindings without type signatures. Hard to grep — check compiler warnings instead.
Run: spago build 2>&1 | grep "No type declaration"
```

**D3. Backward compatibility shims (WARNING)**
```
Pattern: "backward compat|legacy|old API|for compatibility|COMPAT"
Glob: **/*.purs
```

**D4. Dead code markers (INFO)**
```
Pattern: "unused|dead code|no longer used|remove this"
Glob: **/*.purs
```

#### Category E: Project-Specific Rules

Check for a `.claude/fp-police-rules.md` file in the project root. If present, read it and run any additional project-specific checks defined there. This file should follow the same format:

```markdown
# Project-Specific FP Police Rules

## E1. Rule Name (SEVERITY)
Description of what to check.
Pattern: "grep pattern"
Glob: **/*.purs
```

This allows projects to add rules for their own architectural constraints (e.g., "no FFI in demo code", "no direct D3 access outside the selection library") without polluting the general audit.

### 3. Generate the Report

```markdown
# FP Police Audit Report
Date: YYYY-MM-DD
Scope: [full/quick/path]
Project: [current directory name]

## VIOLATIONS (Must Fix)
### [Category] - [Count] issues
- File:Line - Description

## WARNINGS (Should Review)
### [Category] - [Count] issues
- File:Line - Description

## STYLE (Improve)
### [Category] - [Count] issues
- File:Line - Description

## INFO (Track)
### [Category] - [Count] issues
- File:Line - Description

## CLEAN AREAS
Directories that passed with no issues.

## SUMMARY
X violations, Y warnings, Z style issues across N files.

## RECOMMENDATIONS
Prioritized fixes, highest impact first.
```

### 4. After the Audit

1. Output the report
2. Print summary line: "Found X violations, Y warnings, Z style issues across N files"
3. If violations found, ask: "Would you like me to fix any of these?"

## Quick Reference — Grep Patterns

```
# Unsafe operations
unsafeCoerce|unsafePartial|unsafePerformEffect|unsafeCrashWith

# Refs
Effect\.Ref|import.*\bRef\b

# Console
Console\.log|import.*Console

# Partial functions
fromJust|unsafeIndex

# Style: else pure unit
else\s+pure\s+unit

# Internal imports
import.*\.Internal\.

# FFI naming
foreign import (?!.*Impl)

# Deprecated markers
TODO|FIXME|DEPRECATED|HACK|XXX

# String errors
Left ".*"|throwError ".*"
```
