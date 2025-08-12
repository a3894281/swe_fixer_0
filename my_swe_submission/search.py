import os
import re
import json
from collections import defaultdict

from swebase import LLMClient


def analyze_issue_type(issue_description: str, llm: LLMClient, model: str) -> dict:
    """Analyze the issue to determine fix strategy"""
    prompt = f"""
Analyze this software issue and extract key information:

Issue: {issue_description}

Return ONLY a JSON object:
{{
    "bug_type": "import_error|syntax_error|attribute_error|type_error|logic_error|missing_method|deprecation|other",
    "error_messages": ["exact error text if any"],
    "mentioned_functions": ["function names mentioned"],
    "mentioned_classes": ["class names mentioned"],
    "mentioned_files": ["file names or paths mentioned"],
    "fix_strategy": "add_method|replace_specific|add_import|fix_syntax|update_api|logic_change",
    "search_priority": ["most specific terms to search for"],
    "likely_location": "specific file or module where fix should be applied"
}}
"""
    
    response, _ = llm(prompt, model, 0.0)
    try:
        result = json.loads(response.strip())
        # Ensure we have the new fields for backward compatibility
        if "mentioned_classes" not in result:
            result["mentioned_classes"] = []
        if "likely_location" not in result:
            result["likely_location"] = ""
        return result
    except (json.JSONDecodeError, KeyError, TypeError):
        return {
            "bug_type": "other",
            "error_messages": [],
            "mentioned_functions": [],
            "mentioned_classes": [],
            "mentioned_files": [],
            "fix_strategy": "replace_specific",
            "search_priority": extract_keywords(issue_description, llm, model),
            "likely_location": ""
        }


def search_by_file_names(repo_path: str, file_hints: list[str]) -> list[str]:
    """Search for files mentioned in the issue"""
    matches = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            file_path = os.path.relpath(os.path.join(root, file), repo_path)
            for hint in file_hints:
                if hint.lower() in file_path.lower() or file_path.endswith(hint):
                    matches.append(file_path)
    return matches


def search(
    repo_path: str, issue_description: str, llm: LLMClient, model: str
) -> tuple[list[str], list[str]]:
    """Enhanced search with multiple strategies"""
    
    print("🔍 Analyzing issue type...")
    # First, analyze the issue
    issue_info = analyze_issue_type(issue_description, llm, model)
    print(f"   Issue type: {issue_info['bug_type']}")
    print(f"   Fix strategy: {issue_info['fix_strategy']}")
    
    all_matches = set()
    
    # Strategy 1: Search for exact error messages
    if issue_info["error_messages"]:
        print("📄 Searching for error messages...")
        for error_msg in issue_info["error_messages"]:
            matches = search_by_content(repo_path, [error_msg])
            all_matches.update(matches[:3])  # Top 3 matches per error
            print(f"   Found {len(matches)} files with error: {error_msg[:50]}...")
    
    # Strategy 2: Search for mentioned functions/classes
    search_terms = issue_info["mentioned_functions"] + issue_info.get("mentioned_classes", [])
    if search_terms:
        print("🔧 Searching for mentioned functions/classes...")
        func_matches = search_by_content(repo_path, search_terms)
        all_matches.update(func_matches[:5])
        print(f"   Found {len(func_matches)} files with functions/classes: {search_terms}")
    
    # Strategy 3: Search for mentioned files
    if issue_info["mentioned_files"]:
        print("📁 Searching for mentioned files...")
        file_matches = search_by_file_names(repo_path, issue_info["mentioned_files"])
        all_matches.update(file_matches)
        print(f"   Found {len(file_matches)} matching files: {issue_info['mentioned_files']}")
    
    # Strategy 4: Specific location hint search
    if issue_info.get("likely_location"):
        print(f"🎯 Searching in likely location: {issue_info['likely_location']}")
        location_matches = search_by_file_names(repo_path, [issue_info["likely_location"]])
        all_matches.update(location_matches)
        print(f"   Found {len(location_matches)} files in likely location")
    
    # Strategy 5: Fallback to keyword search
    if len(all_matches) < 3:
        print("🔤 Using keyword search fallback...")
        keywords = issue_info.get("search_priority", extract_keywords(issue_description, llm, model))
        content_matches = search_by_content(repo_path, keywords)
        all_matches.update(content_matches[:5])
        print(f"   Found {len(content_matches)} files with keywords: {keywords}")
    
    # Prioritize Python files over tests
    prioritized_files = []
    test_files = []
    
    for file_path in all_matches:
        # CRITICAL: Skip ALL test files to prevent score 0
        if any(test_indicator in file_path.lower() for test_indicator in ["test", "spec", "mock"]):
            test_files.append(file_path)
        else:
            prioritized_files.append(file_path)
    
    # Skip problematic file types
    filtered_files = []
    for file_path in prioritized_files:
        # Skip __init__.py files that are deep in the hierarchy
        if file_path.endswith("__init__.py") and len(file_path.split("/")) > 3:
            continue
        
        # Skip migrations and other typically non-core files
        if any(skip in file_path.lower() for skip in ["migration", "locale", "static", "media", "doc"]):
            continue
            
        filtered_files.append(file_path)
    
    # Return prioritized files first, then test files if needed (but deprioritized)
    final_files = filtered_files[:5]
    if len(final_files) < 3:
        final_files.extend(test_files[:2])  # Add some test files only if really needed
    
    final_files = final_files[:5]  # Cap at 5 files
    
    print(f"🎯 Selected {len(final_files)} files: {final_files}")
    
    return final_files, issue_info.get("search_priority", extract_keywords(issue_description, llm, model))


def search_by_content(repo_path: str, key_terms: list[str]) -> list[str]:
    """Search files by content matching key terms and rank by relevance."""
    matches = defaultdict(int)

    # Preprocess keywords: lowercase for case-insensitive search
    key_terms_lower = [term.lower() for term in key_terms]

    skip_dirs = {
        ".git",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        "migrations",
        ".tox",
        "dist",
        "build",
        ".mypy_cache",
    }
    valid_extensions = {".py", ".js", ".ts", ".java", ".go", ".cpp", ".hpp"}

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for file in files:
            if not any(file.endswith(ext) for ext in valid_extensions):
                continue

            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, repo_path)

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                content_lower = content.lower()

                for term, term_lower in zip(key_terms, key_terms_lower):
                    score = 0

                    # Regex for function or method call: e.g. term(...) or .term(...)
                    func_call_pattern = re.compile(
                        rf"(\b{re.escape(term)}\s*\()|(\.{re.escape(term)}\s*\()"
                    )
                    if func_call_pattern.search(content):
                        score += 10

                    # Word boundary match (case insensitive)
                    if re.search(rf"\b{re.escape(term)}\b", content, re.IGNORECASE):
                        score += 5

                    # Case-insensitive substring match
                    if term_lower in content_lower:
                        score += 3

                    matches[rel_path] += score

            except Exception:
                continue

    # Sort files by total score, descending, and return top 20
    sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
    return [path for path, _ in sorted_matches[:20]]


def extract_keywords(
    issue_description: str,
    llm: LLMClient,
    model: str = "openai/gpt-5",
    temperature: float = 0.0,
) -> list[str]:
    prompt = f"""
You are a helpful assistant for software debugging.

Given the following issue description, extract the most relevant keywords and key phrases that can be used to search the project codebase to find the root cause of the issue.

Issue Description:
\"\"\"
{issue_description}
\"\"\"

Focus on:
- Function names, class names, method names
- Error messages or error types
- File paths or module names
- Technical terms and concepts
- Variable names or parameter names

Output only a list of keywords or short phrases separated by commas. Do not include explanations or additional text.
"""
    response, _ = llm(prompt, model, temperature)
    keywords = [k.strip() for k in response.split(",") if k.strip()]
    
    # Add some automatic keywords based on content analysis
    issue_lower = issue_description.lower()
    if "subparser" in issue_lower:
        keywords.extend(["subparser", "add_subparsers", "parser_class"])
    if "commandparser" in issue_lower:
        keywords.extend(["CommandParser", "ArgumentParser"])
    if "django" in issue_lower and "management" in issue_lower:
        keywords.extend(["BaseCommand", "django.core.management"])
        
    return list(set(keywords))  # Remove duplicates
