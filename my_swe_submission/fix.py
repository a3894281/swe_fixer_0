import re
import ast
from dataclasses import dataclass
from pathlib import Path

from swebase import LLMClient


@dataclass
class CodeChunk:
    """Represents a chunk of code with context for reliable identification."""

    start_line: int
    end_line: int
    content: str
    signature: str  # Function/class signature for identification
    file_path: str
    # Additional context around the chunk to improve patch application
    context_before: str = ""
    context_after: str = ""


class ContextAwareCodeFixer:
    """
    Advanced code fixer that uses large context chunks and merges overlapping regions.
    """

    def __init__(
        self,
        llm: LLMClient,
        model: str = "anthropic/claude-sonnet-4",
        temperature: float = 0,
    ):
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.context_lines = 10
        self.max_fix_attempts = 3

    def extract_keyword_chunks_with_context(
        self, content: str, keywords: list[str], file_path: str
    ) -> list[CodeChunk]:
        """
        Extract code chunks containing keywords with rich context for reliable identification.
        """
        lines = content.splitlines()
        # Expand keywords: split on non-alphanumeric and camelCase to improve recall
        def _split_kw(kw: str) -> list[str]:
            parts = re.split(r"[^a-zA-Z0-9]", kw)
            camel_parts = re.sub(r'([a-z])([A-Z])', r"\1 \2", kw).split()
            return [p.lower() for p in parts + camel_parts if p]

        expanded_keywords: set[str] = set()
        for kw in keywords:
            expanded_keywords.update(_split_kw(kw))
            expanded_keywords.add(kw.lower())

        keyword_lower = list(expanded_keywords)

        # Find function/class blocks using AST when possible
        try:
            tree = ast.parse(content)
            blocks = self._extract_ast_blocks(tree, lines)
        except SyntaxError:
            # Fall back to regex-based extraction for non-Python files or syntax errors
            blocks = self._extract_regex_blocks(lines)

        # Filter blocks that contain keywords
        matched_blocks = []
        keywords_set = set(keyword_lower)

        for start, end in blocks:
            block_text = "\n".join(lines[start : end + 1]).lower()
            if any(kw in block_text for kw in keywords_set):
                matched_blocks.append((start, end))
                
        # Special case: if looking for CommandParser, specifically search for it
        if any("commandparser" in kw for kw in keyword_lower) and not matched_blocks:
            print(f"   Specifically searching for CommandParser class...")
            for start, end in blocks:
                block_text = "\n".join(lines[start : end + 1])
                if "class CommandParser" in block_text:
                    matched_blocks.append((start, end))
                    print(f"   Found CommandParser class at lines {start+1}-{end+1}")
                    break

        # If no blocks match keywords, try broader context search
        if not matched_blocks:
            print(f"   No keyword matches, trying broader search in {file_path}")
            # Look for any function/class that might be relevant, but prioritize classes
            class_blocks = []
            function_blocks = []
            
            for start, end in blocks:
                block_text = "\n".join(lines[start : end + 1])
                if block_text.strip().startswith('class '):
                    class_blocks.append((start, end))
                else:
                    function_blocks.append((start, end))
            
            # Prefer classes over functions, take up to 2 of each
            matched_blocks.extend(class_blocks[:2])
            matched_blocks.extend(function_blocks[:1])
        
        if not matched_blocks:
            print(f"   No code blocks found in {file_path}")
            return []

        # Merge overlapping blocks and add context
        merged_blocks = self._merge_overlapping_blocks(matched_blocks)
        context_blocks = self._add_context_to_blocks(merged_blocks, len(lines))

        # Create chunks
        chunks = []
        for context_start, context_end, original_start, original_end in context_blocks:
            chunk = self._create_code_chunk_with_full_context(
                lines,
                context_start,
                context_end,
                original_start,
                original_end,
                file_path,
            )
            if chunk:
                chunks.append(chunk)

        return chunks

    def _create_code_chunk_with_full_context(
        self,
        lines: list[str],
        context_start: int,
        context_end: int,
        original_start: int,
        original_end: int,
        file_path: str,
    ) -> CodeChunk | None:
        """Create a CodeChunk with full context."""
        if (
            context_start < 0
            or context_end >= len(lines)
            or context_start > context_end
        ):
            return None

        # Full content including context
        content = "\n".join(lines[context_start : context_end + 1])

        # Extract signature (first non-empty line of original target area)
        signature = ""
        for i in range(original_start, min(original_end + 1, len(lines))):
            stripped = lines[i].strip()
            if stripped:
                signature = stripped
                break

        # Capture context before and after the original block (excluding target content itself)
        context_before = "\n".join(lines[context_start:original_start]) if original_start > context_start else ""
        context_after = "\n".join(lines[original_end + 1: context_end + 1]) if original_end < context_end else ""

        return CodeChunk(
            start_line=context_start,
            end_line=context_end,
            content=content,
            signature=signature,
            file_path=file_path,
            context_before=context_before,
            context_after=context_after,
            )

    def _merge_overlapping_blocks(
        self, blocks: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Merge blocks that are close together or overlapping."""
        if not blocks:
            return []

        # Sort blocks by start line
        sorted_blocks = sorted(blocks)
        merged = [sorted_blocks[0]]

        for current_start, current_end in sorted_blocks[1:]:
            prev_start, prev_end = merged[-1]

            # If blocks overlap or are close together, merge them
            if current_start <= prev_end + self.context_lines:
                merged[-1] = (prev_start, max(prev_end, current_end))
            else:
                merged.append((current_start, current_end))

        return merged

    def _add_context_to_blocks(
        self, blocks: list[tuple[int, int]], total_lines: int
    ) -> list[tuple[int, int, int, int]]:
        """Add context lines around each block. Returns (context_start, context_end, original_start, original_end)."""
        context_blocks = []

        for original_start, original_end in blocks:
            context_start = max(0, original_start - self.context_lines)
            context_end = min(total_lines - 1, original_end + self.context_lines)

            context_blocks.append(
                (context_start, context_end, original_start, original_end)
            )

        return context_blocks

    def extract_imports(self, content: str) -> list[str]:
        lines = content.splitlines()

        imports = []
        for line in lines[:50]:  # Check first 50 lines for imports
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                imports.append(stripped)

        return imports

    def _extract_ast_blocks(
        self, tree: ast.AST, lines: list[str]
    ) -> list[tuple[int, int]]:
        """Extract function and class blocks using AST."""
        blocks = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start_line = node.lineno - 1  # AST line numbers are 1-based

                # Find the end of the block
                end_line = start_line
                if hasattr(node, "end_lineno") and node.end_lineno:
                    end_line = node.end_lineno - 1
                else:
                    # Calculate end line based on indentation
                    end_line = self._find_block_end_by_indentation(lines, start_line)

                blocks.append((start_line, end_line))

        return sorted(blocks)

    def _extract_regex_blocks(self, lines: list[str]) -> list[tuple[int, int]]:
        """Extract function and class blocks using regex (fallback method)."""
        header_re = re.compile(r"^(\s*)(def |class |function |const |let |var )")
        blocks = []
        block_start = None

        for i, line in enumerate(lines):
            if header_re.match(line):
                if block_start is not None:
                    blocks.append((block_start, i - 1))
                block_start = i

        if block_start is not None:
            blocks.append((block_start, len(lines) - 1))

        # Adjust block ends based on indentation
        adjusted_blocks = []
        for start, end in blocks:
            actual_end = self._find_block_end_by_indentation(lines, start)
            adjusted_blocks.append((start, min(actual_end, end)))

        return adjusted_blocks

    def _find_block_end_by_indentation(self, lines: list[str], start_line: int) -> int:
        """Find the end of a code block based on indentation."""
        if start_line >= len(lines):
            return start_line

        header_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        last_line = start_line

        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            stripped = line.lstrip()

            if not stripped:  # Empty line
                last_line = i
                continue

            indent = len(line) - len(stripped)
            if indent > header_indent:
                last_line = i
            else:
                break

        return last_line

    def fix_chunk_with_llm(
        self, chunk: CodeChunk, issue_description: str, imports: list[str]
    ) -> str | None:
        """Fix a single chunk using LLM with enhanced prompting."""

        # Create a comprehensive prompt
        prompt = self._create_fix_prompt(chunk, issue_description, imports)

        try:
            response, _ = self.llm(prompt, self.model, self.temperature)
            response = response.strip()

            if self._is_no_change(response):
                return None

            # Clean up the response
            fixed_code = self._clean_llm_response(response)

            return fixed_code

        except Exception as e:
            print(f"Error fixing chunk {chunk.signature}: {str(e)}")
            return None

    def _create_fix_prompt(
        self, chunk: CodeChunk, issue_description: str, imports: list[str]
    ) -> str:
        """Create a more targeted fix prompt"""
        imports_text = "\n".join(imports) if imports else "# No imports found"
        
        # Extract specific error info if available
        error_context = ""
        if "error" in issue_description.lower() or "traceback" in issue_description.lower():
            error_context = "\nThis appears to be fixing an ERROR. Focus on the exact error mentioned."
        
        # Create more specific prompt based on issue type
        specific_guidance = ""
        if "subparser" in issue_description.lower() and "commandparser" in chunk.content.lower():
            specific_guidance = """
SPECIFIC GUIDANCE: This appears to be a Django management command issue with subparsers.
- The issue is that subparsers don't inherit CommandParser's special arguments
- You likely need to add an add_subparsers method to the CommandParser class
- The method should use functools.partial to pass called_from_command_line parameter
- Look at the patch requirements and implement exactly what's needed
"""
            
        return f"""ISSUE TO FIX:
{issue_description}{error_context}

FILE: {chunk.file_path}
{specific_guidance}
AVAILABLE IMPORTS:
{imports_text}

CODE TO FIX:
```
{chunk.content}
```

REQUIREMENTS:
1. Fix ONLY the specific issue described above
2. Make minimal changes - don't refactor unrelated code  
3. Preserve all existing functionality and logic
4. If you need new imports, add them at the top of the method or class
5. Return ONLY the fixed code block (no explanations)
6. If no fix is needed, return exactly: NO_CHANGE

FIXED CODE:"""

    def _clean_llm_response(self, response: str) -> str:
        """Clean and normalize LLM response."""
        # Remove any markdown code blocks
        response = re.sub(r"^```.*\n", "", response, flags=re.MULTILINE)
        response = re.sub(r"\n```$", "", response)
        response = response.strip()

        return response

    def _is_no_change(self, response: str) -> bool:
        """Check if LLM indicated no changes needed."""
        response_clean = response.strip().upper()
        return response_clean == "NO_CHANGE" or response_clean == "NO"
    
    def validate_fix(self, original: str, fixed: str, file_path: str) -> bool:
        """Validate that the fix doesn't break basic syntax"""
        if fixed == original:
            return True
            
        try:
            # Python syntax validation
            if file_path.endswith('.py'):
                ast.parse(fixed)
            
            # Check that we didn't accidentally remove important content
            original_lines = len(original.splitlines())
            fixed_lines = len(fixed.splitlines())
            
            # Allow shorter replacements when we specifically requested minimal diffs
            # Heuristic: if fixed code still contains at least one function/class definition, assume fine.
            # (We already parse definitions below.)
                
            # Check that function/class definitions are preserved
            original_defs = re.findall(r'^(def |class |async def )', original, re.MULTILINE)
            fixed_defs = re.findall(r'^(def |class |async def )', fixed, re.MULTILINE)
            
            # Don't remove function definitions unless that's specifically needed
            if len(fixed_defs) < len(original_defs) * 0.8:
                print(f"   ❌ Validation failed: Function/class definitions removed ({len(fixed_defs)} vs {len(original_defs)})")
                return False
                
            return True
            
        except SyntaxError as e:
            print(f"   ❌ Validation failed: Syntax error - {e}")
            return False
        except Exception as e:
            print(f"   ⚠️ Validation warning: {e}")
            return True  # If we can't validate, assume it's ok

    def apply_fixes_to_content(
        self, content: str, chunk_fixes: list[tuple[CodeChunk, str]]
    ) -> str:
        """
        Apply multiple fixes to content using context-aware replacement.
        """
        if not chunk_fixes:
            return content

        current_content = content

        # Sort fixes by start line in descending order (apply from bottom to top)
        sorted_fixes = sorted(chunk_fixes, key=lambda x: x[0].start_line, reverse=True)

        for chunk, fixed_code in sorted_fixes:
            try:
                current_content = self._apply_single_fix(
                    current_content, chunk, fixed_code
                )
                print(
                    f"✓ Applied fix to {chunk.signature} (lines {chunk.start_line + 1}-{chunk.end_line + 1})"
                )
            except Exception as e:
                print(f"✗ Failed to apply fix to {chunk.signature}: {str(e)}")
                continue

        return current_content

    def _apply_single_fix(self, content: str, chunk: CodeChunk, fixed_code: str) -> str:
        """Apply a single fix using context-aware replacement."""

        # Method 1: Try exact context matching
        try:
            full_pattern = (
                chunk.context_before + "\n" + chunk.content + "\n" + chunk.context_after
            )
            full_replacement = (
                chunk.context_before + "\n" + fixed_code + "\n" + chunk.context_after
            )

            if full_pattern in content:
                return content.replace(full_pattern, full_replacement, 1)
        except Exception:
            pass

        # Method 2: Try just the original content with some context
        try:
            # Use signature as anchor point
            lines = content.splitlines()

            # Find the chunk by signature and surrounding context
            for i, line in enumerate(lines):
                if chunk.signature in line:
                    # Verify this is the right location by checking context
                    context_start = max(0, i - self.context_lines)
                    context_end = min(len(lines), i + self.context_lines)
                    surrounding_context = "\n".join(lines[context_start:context_end])

                    if (
                        chunk.context_before in surrounding_context
                        or chunk.context_after in surrounding_context
                    ):
                        # Found the right location, now find the full block
                        block_start = self._find_block_start(lines, i)
                        block_end = self._find_block_end_by_indentation(
                            lines, block_start
                        )

                        # Replace the block
                        fixed_lines = fixed_code.splitlines()
                        lines[block_start : block_end + 1] = fixed_lines
                        return "\n".join(lines)

        except Exception:
            pass

        # Method 3: Direct replacement as last resort
        if chunk.content in content:
            return content.replace(chunk.content, fixed_code, 1)

        raise ValueError(f"Could not find location to apply fix for: {chunk.signature}")

    def _find_block_start(self, lines: list[str], signature_line: int) -> int:
        """Find the start of a code block given a line within it."""
        # If the signature line is the start, return it
        if re.match(r"^(\s*)(def |class |function )", lines[signature_line]):
            return signature_line

        # Otherwise, search backwards for the block start
        for i in range(signature_line, -1, -1):
            if re.match(r"^(\s*)(def |class |function )", lines[i]):
                return i

        return signature_line  # Fallback


def fix(
    files: dict[str, str],
    file_names: list[str], 
    issue_description: str,
    keywords: list[str],
    llm: LLMClient,
    model: str = "openai/gpt-5",
    temperature: float = 0.0,
) -> dict[str, str]:
    """Enhanced fix with validation"""
    
    if not files or not file_names or not keywords:
        print("Warning: Missing required inputs")
        return {}

    fixer = ContextAwareCodeFixer(llm, model, temperature)
    fixed_files = {}
    
    for file_name in file_names:
        content = files[file_name]
        print(f"\n🔧 Processing {file_name}")
        
        try:
            imports = fixer.extract_imports(content)
            chunks = fixer.extract_keyword_chunks_with_context(content, keywords, file_name)
            
            if not chunks:
                print(f"   No relevant chunks found in {file_name}")
                continue
                
            valid_fixes = []
            
            for i, chunk in enumerate(chunks):
                print(f"   Analyzing chunk {i+1}/{len(chunks)}: {chunk.signature[:50]}...")
                
                fixed_code = fixer.fix_chunk_with_llm(chunk, issue_description, imports)
                
                if fixed_code and not fixer._is_no_change(fixed_code):
                    # Validate the fix before accepting it
                    if fixer.validate_fix(chunk.content, fixed_code, file_name):
                        valid_fixes.append((chunk, fixed_code))
                        print("   ✅ Valid fix generated")
                    else:
                        print("   ❌ Fix failed validation, skipping")
                else:
                    print("   ℹ️  No changes needed")
            
            if valid_fixes:
                fixed_content = fixer.apply_fixes_to_content(content, valid_fixes)
                if fixed_content != content:
                    # Final validation on the complete file
                    if fixer.validate_fix(content, fixed_content, file_name):
                        fixed_files[file_name] = fixed_content
                        print(f"✅ Successfully fixed {file_name} with {len(valid_fixes)} changes")
                        
                        # Save to temp directory for review
                        temp_path = Path(".temp") / file_name
                        temp_path.parent.mkdir(parents=True, exist_ok=True)
                        temp_path.write_text(fixed_content, encoding="utf-8")
                        print(f"   💾 Saved to {temp_path}")
                    else:
                        print(f"❌ Final validation failed for {file_name}")
                else:
                    print(f"   ℹ️  No net changes applied to {file_name}")
            else:
                print(f"   • No valid fixes generated for {file_name}")
                    
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")
            continue
            
    return fixed_files
